"""Smart Parking - single-file Flask app (video -> YOLO -> slots -> billing -> dashboard).

Run:     python app.py        (dev)
         AUTOSTART=1 gunicorn -w 1 --threads 8 -b 0.0.0.0:5000 app:app     (production; keep exactly ONE worker)
Config:  a .env file (KEY=value lines), environment variables, or an optional config.json; real env vars win over both files.
  Pricing    RATE, FREE_SECS, MIN_CHARGE, DAILY_CAP (per slot), PEAK_START, PEAK_END, PEAK_MULT, BUS_MULT, TRUCK_MULT
  Detection  PARK_SECS, OCC_GRACE, CONF, YOLO_MODEL, TRACKER (bytetrack | botsort | empty = built-in IoU tracker)
  Video      VIDEO_SOURCE (webcam index, rtsp:// or http:// URL; empty = first file in ./videos)
  Behaviour  AUTO_SLOTS=0 (no automatic slot discovery), WALLET=1 (prepaid wallet), LOG_EVERY (history interval, s)
  Security   SECRET_KEY, PUBLIC_URL (https => secure cookies), BEHIND_PROXY=1, WEBHOOK_SECRET
  Email      SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, SMTP_FROM, ADMIN_EMAIL
The first account that registers becomes the admin.
"""
import os, glob, time, sqlite3, secrets, threading, csv, io, json, hmac, smtplib
from email.message import EmailMessage
import cv2, requests
from flask import Flask, request, session, redirect, jsonify, Response, render_template_string
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.middleware.proxy_fix import ProxyFix

def load_dotenv(path=".env"):
    """Tiny .env reader: KEY=value per line, # comments, optional quotes and 'export ' prefix. Real env vars win."""
    try: lines = open(path, encoding="utf-8-sig").read().splitlines()
    except OSError: return
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line: continue
        k, v = line.split("=", 1); k = k.strip()
        if k.startswith("export "): k = k[7:].strip()
        v = v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'": v = v[1:-1]
        elif " #" in v: v = v.split(" #", 1)[0].rstrip()          # inline comment on unquoted values
        if k: os.environ.setdefault(k, v)
load_dotenv()

try:    # optional config.json supplies defaults for the variables above
    for _k, _v in json.load(open("config.json")).items(): os.environ.setdefault(str(_k), str(_v))
except (OSError, ValueError, AttributeError): pass

def env_f(k, d):
    try: return float(os.getenv(k, d))
    except ValueError: return float(d)

RATE = int(env_f("RATE", 10))                  # rupees per second (video time)
FREE_SECS = env_f("FREE_SECS", 0)              # first N seconds are free
MIN_CHARGE = env_f("MIN_CHARGE", 0)            # minimum charge once anything is billed
DAILY_CAP = env_f("DAILY_CAP", 0)              # max rupees per slot per day (0 = no cap)
PEAK_START, PEAK_END, PEAK_MULT = int(env_f("PEAK_START", 18)), int(env_f("PEAK_END", 21)), env_f("PEAK_MULT", 1)
VTYPE_MULT = {"car": 1.0, "bus": env_f("BUS_MULT", 2), "truck": env_f("TRUCK_MULT", 2)}
VT_NAMES = {2: "car", 5: "bus", 7: "truck"}    # COCO class id -> vehicle type
WALLET = os.getenv("WALLET") == "1"
AUTO_SLOTS = os.getenv("AUTO_SLOTS", "1") != "0"
TRACKER = os.getenv("TRACKER", "").strip().lower() or None
VIDEO_SOURCE = (os.getenv("VIDEO_SOURCE") or "").strip() or None
LOG_EVERY = env_f("LOG_EVERY", 30)
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET")
SMTP_HOST, SMTP_PORT = os.getenv("SMTP_HOST"), int(env_f("SMTP_PORT", 587))
SMTP_USER, SMTP_PASS = os.getenv("SMTP_USER"), os.getenv("SMTP_PASS")
SMTP_FROM = os.getenv("SMTP_FROM") or SMTP_USER or "parking@localhost"
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL")
VIDEO_DIR = "videos"
TF_KEY = os.getenv("TINYFISH_API_KEY")
PUBLIC_URL = os.getenv("PUBLIC_URL", "http://localhost:5000").rstrip("/")
PAYMENT_URL = os.getenv("PAYMENT_URL")   # public test payment page (no tunnel needed)
DB = "parking.db"

def load_secret():
    """SECRET_KEY from env, else a random key generated once and kept in .secret_key (never a guessable default)."""
    k = os.getenv("SECRET_KEY")
    if k: return k
    try:
        k = open(".secret_key").read().strip()
        if k: return k
    except OSError: pass
    k = secrets.token_hex(32)
    try: open(".secret_key", "w").write(k)
    except OSError: pass
    return k

app = Flask(__name__); app.secret_key = load_secret()
app.config.update(SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax", SESSION_COOKIE_SECURE=PUBLIC_URL.startswith("https"))
if os.getenv("BEHIND_PROXY") == "1": app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1)

def db():
    c = sqlite3.connect(DB); c.row_factory = sqlite3.Row; return c
with db() as c:
    c.executescript("""
    CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY, name TEXT, email TEXT UNIQUE, pw TEXT);
    CREATE TABLE IF NOT EXISTS bookings(id INTEGER PRIMARY KEY, user_id INT, slot INT, start REAL, end REAL, amount INT);
    CREATE TABLE IF NOT EXISTS charges(id INTEGER PRIMARY KEY, slot INT, seconds INT, amount INT,
        status TEXT, token TEXT, run TEXT, created REAL);
    CREATE TABLE IF NOT EXISTS slots(id INTEGER PRIMARY KEY, x1 INT, y1 INT, x2 INT, y2 INT);
    CREATE TABLE IF NOT EXISTS occ_log(ts REAL, free INT, occupied INT, booked INT);
    CREATE TABLE IF NOT EXISTS parkings(id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, address TEXT, lat REAL, lng REAL, live INT DEFAULT 0);""")
    for _sql in ("ALTER TABLE users ADD COLUMN role TEXT DEFAULT 'user'", "ALTER TABLE users ADD COLUMN balance REAL DEFAULT 0",
                 "ALTER TABLE charges ADD COLUMN vtype TEXT DEFAULT 'car'"):
        try: c.execute(_sql)
        except sqlite3.OperationalError: pass          # column already there
    if not c.execute("SELECT 1 FROM users WHERE role='admin'").fetchone():     # upgrade: oldest account becomes admin
        c.execute("UPDATE users SET role='admin' WHERE id=(SELECT MIN(id) FROM users)")
    if not c.execute("SELECT 1 FROM parkings").fetchone():     # first run: the parking that has your video + a few demo ones
        _lat, _lng = env_f("LIVE_LAT", 28.6692), env_f("LIVE_LNG", 77.4538)    # set LIVE_LAT / LIVE_LNG (or edit it in the admin page)
        c.execute("INSERT INTO parkings(name,address,lat,lng,live) VALUES(?,?,?,?,1)",
                  (os.getenv("LIVE_NAME", "Smart Parking (live camera)"), os.getenv("LIVE_ADDRESS", ""), _lat, _lng))
        for _n, _dl, _dg in (("City Centre Parking (demo)", 0.009, 0.006), ("Station Road Parking (demo)", -0.014, 0.011), ("Mall Parking (demo)", 0.021, -0.017)):
            c.execute("INSERT INTO parkings(name,address,lat,lng,live) VALUES(?,?,?,?,0)", (_n, "Demo entry - no live camera", _lat + _dl, _lng + _dg))

# ---------------- vision / tracking ----------------
S = {"slots": {}, "occ": {}, "since": {}, "last_seen": {}, "vtype": {}, "vt": 0.0, "status": "starting", "jpg": None,
     "next_id": 0, "auto": AUTO_SLOTS, "fps": 0, "det_ms": 0, "started": time.time()}
LOCK = threading.RLock()
with db() as _c:                                   # slots survive restarts
    for _r in _c.execute("SELECT * FROM slots"): S["slots"][_r["id"]] = (_r["x1"], _r["y1"], _r["x2"], _r["y2"])
S["next_id"] = max(S["slots"], default=0)

def add_slot(box):
    with LOCK:
        S["next_id"] += 1; sid = S["next_id"]; b = tuple(int(v) for v in box); S["slots"][sid] = b
        with db() as c: c.execute("INSERT OR REPLACE INTO slots VALUES(?,?,?,?,?)", (sid, *b))
    return sid

def move_slot(sid, box):
    with LOCK:
        if sid not in S["slots"]: return False
        b = tuple(int(v) for v in box); S["slots"][sid] = b
        with db() as c: c.execute("UPDATE slots SET x1=?,y1=?,x2=?,y2=? WHERE id=?", (*b, sid))
    return True

def drop_slot(sid):
    with LOCK:
        if sid not in S["slots"]: return False
        if S["occ"].get(sid):                      # bill a car that is still there before the slot disappears
            new_charge(sid, max(0, S["last_seen"].get(sid, S["vt"]) - S["since"].get(sid, S["vt"])), S["vtype"].get(sid, "car"))
        S["slots"].pop(sid)
        for k in ("occ", "since", "last_seen", "vtype"): S[k].pop(sid, None)
        with db() as c: c.execute("DELETE FROM slots WHERE id=?", (sid,))
    return True

def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x2 - x1) * max(0, y2 - y1)
    u = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i
    return i / u if u else 0

PARK_SECS = env_f("PARK_SECS", 3.0)    # a car must stand still this long (video seconds) to count as parked
OCC_GRACE = env_f("OCC_GRACE", 2.5)    # seconds a parked car may be missed by the detector before the slot is freed
CONF = env_f("CONF", 0.3)
YOLO_MODEL = os.getenv("YOLO_MODEL", "yolov8s.pt")   # try yolov8m.pt for more accuracy (slower)
SH = {"frame": None, "vt": 0.0, "seq": 0, "pass_end": None}   # player -> detector hand-off

def still(a, b):
    """True if box b is the same car as a that has not moved (centre drift + size change tiny)."""
    wa, ha = a[2]-a[0], a[3]-a[1]; d = max(wa, ha) * 0.12
    return (abs((a[0]+a[2]) - (b[0]+b[2])) / 2 <= d and abs((a[1]+a[3]) - (b[1]+b[3])) / 2 <= d
            and abs((b[2]-b[0]) - wa) <= 0.25 * wa and abs((b[3]-b[1]) - ha) <= 0.25 * ha)
def med(v): v = sorted(v); return v[len(v) // 2]

def layout():
    """Snap slots into a symmetric grid: equal size, shared row lines and column lines.
    Returns ({slot_id: (x1,y1,x2,y2)}, rows) with rows ordered top->bottom, left->right."""
    sl = S["slots"]
    if not sl: return {}, []
    w, h = med([b[2]-b[0] for b in sl.values()]), med([b[3]-b[1] for b in sl.values()])
    cy = lambda b: (b[1] + b[3]) / 2
    items = sorted(sl.items(), key=lambda kv: cy(kv[1]))
    rows, cur = [], [items[0]]
    for it in items[1:]:
        if abs(cy(it[1]) - sum(cy(b) for _, b in cur) / len(cur)) <= h * 0.5: cur.append(it)
        else: rows.append(cur); cur = [it]
    rows.append(cur)
    xs = sorted(((b[0] + b[2]) / 2, sid) for sid, b in sl.items())
    cols, cc = [], [xs[0]]
    for x, sid in xs[1:]:
        if x - sum(v for v, _ in cc) / len(cc) <= w * 0.5: cc.append((x, sid))
        else: cols.append(cc); cc = [(x, sid)]
    cols.append(cc)
    colx = {sid: med([v for v, _ in c]) for c in cols for _, sid in c}
    out, order = {}, []
    for r in rows:
        y = med([cy(b) for _, b in r]); r = sorted(r, key=lambda kv: colx[kv[0]])
        order.append([sid for sid, _ in r])
        for sid, _ in r:
            out[sid] = (int(colx[sid] - w / 2), int(y - h / 2), int(colx[sid] + w / 2), int(y + h / 2))
    return out, order

def grid_order(): return layout()[1]

def inside(b, x, y, pad=0.1):
    pw, ph = (b[2]-b[0]) * pad, (b[3]-b[1]) * pad
    return b[0]-pw <= x <= b[2]+pw and b[1]-ph <= y <= b[3]+ph

def booked_now(slot):
    t = time.time()
    with db() as c:
        return c.execute("SELECT 1 FROM bookings WHERE slot=? AND start<=? AND end>?", (slot, t, t)).fetchone() is not None

def is_peak():
    h = time.localtime().tm_hour
    if PEAK_MULT == 1 or PEAK_START == PEAK_END: return False
    return PEAK_START <= h < PEAK_END if PEAK_START < PEAK_END else (h >= PEAK_START or h < PEAK_END)

def price(seconds, vtype="car", slot=None):
    """Rupees for a stay: free seconds, vehicle multiplier, peak multiplier, minimum charge, per-slot daily cap."""
    secs = max(0, int(round(seconds)) - int(FREE_SECS))
    amt = int(round(secs * RATE * VTYPE_MULT.get(vtype, 1.0) * (PEAK_MULT if is_peak() else 1)))
    if amt > 0 and MIN_CHARGE: amt = max(amt, int(MIN_CHARGE))
    if amt > 0 and DAILY_CAP and slot is not None:
        midnight = time.mktime(time.localtime()[:3] + (0, 0, 0, 0, 0, -1))
        with db() as c:
            done = c.execute("SELECT COALESCE(SUM(amount),0) FROM charges WHERE slot=? AND created>=?", (slot, midnight)).fetchone()[0]
        amt = max(0, min(amt, int(DAILY_CAP) - int(done)))
    return amt

def pricing_text():
    p = [f"\u20b9{RATE} per second parked"]
    extra = [f"{k} x{v:g}" for k, v in VTYPE_MULT.items() if k != "car" and v != 1]
    if extra: p.append(", ".join(extra))
    if FREE_SECS: p.append(f"first {FREE_SECS:g}s free")
    if MIN_CHARGE: p.append(f"min \u20b9{MIN_CHARGE:g}")
    if DAILY_CAP: p.append(f"daily cap \u20b9{DAILY_CAP:g} per slot")
    if PEAK_MULT != 1: p.append(f"peak {PEAK_START}-{PEAK_END}h x{PEAK_MULT:g}")
    return " \u00b7 ".join(p)

def notify(subject, body, to=None):
    """Optional e-mail alert (needs SMTP_HOST). Never blocks or raises."""
    to = to or ADMIN_EMAIL
    if not (SMTP_HOST and to): return
    def go():
        try:
            m = EmailMessage(); m["Subject"], m["From"], m["To"] = subject, SMTP_FROM, to; m.set_content(body)
            with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=15) as s:
                try: s.starttls()
                except smtplib.SMTPException: pass
                if SMTP_USER: s.login(SMTP_USER, SMTP_PASS or "")
                s.send_message(m)
        except Exception as e: print("email failed:", e)
    threading.Thread(target=go, daemon=True).start()

def paid_alert(cid):
    with db() as c: ch = c.execute("SELECT * FROM charges WHERE id=?", (cid,)).fetchone()
    if ch: notify(f"Parking payment #{cid}", f"Slot {ch['slot']} ({ch['vtype'] or 'car'}), {ch['seconds']} s, \u20b9{ch['amount']} - status {ch['status']}.")

def new_charge(slot, seconds, vtype="car"):
    amt = price(seconds, vtype, slot)
    if amt <= 0: return
    tok = secrets.token_urlsafe(12)
    with db() as c:
        cid = c.execute("INSERT INTO charges(slot,seconds,amount,status,token,created,vtype) VALUES(?,?,?,?,?,?,?)",
                        (slot, int(round(seconds)), amt, "PENDING", tok, time.time(), vtype)).lastrowid
    threading.Thread(target=tinyfish_pay, args=(cid, tok, amt), daemon=True).start()

def tinyfish_pay(cid, tok, amt):
    """Use a TinyFish web agent to pay the charge on the (fake) payment page."""
    if not TF_KEY:  # fallback simulation
        time.sleep(2); settle(cid, tok, "SIMULATED"); return
    if PAYMENT_URL:   # agent pays on an existing PUBLIC test payment page
        goal = (f"This is a TEST-MODE payment page (no real money). If it asks for an amount, enter {amt}. "
                f"Fill any required fields with: name Test User, email test@example.com, phone 9999999999. "
                f"Pay with test card 4242 4242 4242 4242, expiry 12/30, CVV 123 (if OTP/3D-secure appears use the test success option). "
                f"Submit the payment and wait for a success confirmation. "
                f"Return JSON {{\"paid\": true}} only if success is shown, else {{\"paid\": false}}.")
        try:
            r = requests.post("https://agent.tinyfish.ai/v1/automation/run", timeout=600,
                headers={"X-API-Key": TF_KEY, "Content-Type": "application/json"},
                json={"url": PAYMENT_URL, "goal": goal}).json()
            res = r.get("result") or {}
            ok = r.get("status") == "COMPLETED" and isinstance(res, dict) and res.get("paid") is True
            with db() as c:
                c.execute("UPDATE charges SET status=?, run=? WHERE id=?",
                          ("PAID" if ok else "FAILED", str(r.get("run_id") or r.get("status")), cid))
            if ok: paid_alert(cid)
        except Exception as e:
            with db() as c: c.execute("UPDATE charges SET status='FAILED', run=? WHERE id=?", (str(e)[:80], cid))
        return
    goal = (f"This is a demo payment page. Enter card number 4242 4242 4242 4242, expiry 12/30, CVV 123, "
            f"confirm the amount is {amt} rupees, click the 'Pay Now' button, wait for the 'Payment successful' message. "
            f"Return JSON {{\"paid\": true}} if you see it, else {{\"paid\": false}}.")
    try:
        r = requests.post("https://agent.tinyfish.ai/v1/automation/run", timeout=300,
            headers={"X-API-Key": TF_KEY, "Content-Type": "application/json"},
            json={"url": f"{PUBLIC_URL}/pay/{cid}?t={tok}", "goal": goal}).json()
        with db() as c:
            c.execute("UPDATE charges SET run=? WHERE id=?", (str(r.get("run_id") or r.get("status")), cid))
    except Exception as e:
        with db() as c: c.execute("UPDATE charges SET status='FAILED', run=? WHERE id=?", (str(e)[:80], cid))

def settle(cid, tok, run=None):
    with db() as c:
        n = c.execute("UPDATE charges SET status='PAID', run=COALESCE(?,run) WHERE id=? AND token=?", (run, cid, tok)).rowcount
    if n: paid_alert(cid)

def process(frame, vt, state):
    """Detect cars; only cars that stay STILL for PARK_SECS are 'parked'. Slots come only from parked cars in line."""
    tracks = state["tracks"]
    kw = dict(classes=[2, 5, 7], imgsz=960, conf=CONF, iou=0.5, verbose=False)
    res, byte = None, False
    if TRACKER and not S.get("tracker_err"):        # optional ByteTrack / BoT-SORT; falls back to the IoU tracker on any problem
        try: res = state["model"].track(frame, persist=True, tracker=TRACKER + ".yaml", **kw); byte = True
        except Exception as e: S["tracker_err"] = repr(e)[:80]; print("tracker unavailable, using IoU tracker:", e)
    if res is None: res = state["model"](frame, **kw)
    dets = []
    for r in res:
        for bx in r.boxes:
            x1, y1, x2, y2 = map(int, bx.xyxy[0].tolist()); w, h = x2 - x1, y2 - y1
            if min(w, h) >= 12 and 0.25 <= w / max(h, 1) <= 4:
                bid = int(bx.id[0]) if byte and getattr(bx, "id", None) is not None else None
                dets.append(((x1, y1, x2, y2), VT_NAMES.get(int(bx.cls[0]), "car"), bid))
    used = set()
    for d, vtype, bid in dets:
        best, bi = None, 0.3
        if bid is not None: best = ("b", bid)        # id supplied by ByteTrack / BoT-SORT
        else:                                        # built-in IoU tracker
            for t, tr in tracks.items():
                if t not in used and iou(tr["box"], d) > bi: best, bi = t, iou(tr["box"], d)
        if best is None or best not in tracks:
            if best is None: state["tid"] += 1; best = state["tid"]
            tracks[best] = {"box": d, "anchor": d, "avg": d, "anchor_vt": vt, "seen": vt, "cls": vtype}
        tr = tracks[best]; tr["box"], tr["seen"], tr["cls"] = d, vt, vtype; used.add(best)
        if still(tr["anchor"], d): tr["avg"] = tuple(0.85 * p + 0.15 * q for p, q in zip(tr["avg"], d))
        else: tr["anchor"], tr["avg"], tr["anchor_vt"] = d, d, vt          # it moved -> restart the still timer
    for t in [t for t, tr in tracks.items() if vt - tr["seen"] > 2.0]: del tracks[t]
    cur = [tr for tr in tracks.values() if tr["seen"] == vt]
    parked = [tr for tr in cur if vt - tr["anchor_vt"] >= PARK_SECS]
    ctr = lambda b: ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
    with LOCK:
        for tr in parked:                            # new slot: parked + in line with another parked car / existing row
            ab = tuple(int(v) for v in tr["avg"]); cx, cy = ctr(ab); hh = ab[3] - ab[1]
            lay, _ = layout()
            if any(inside(l, cx, cy, 0) for l in lay.values()): continue
            mates = [p for p in parked if p is not tr and abs(ctr(p["avg"])[1] - cy) <= hh * 0.5]
            row_slots = [l for l in lay.values() if abs((l[1] + l[3]) / 2 - cy) <= hh * 0.5]
            if S["auto"] and (mates or row_slots): add_slot(ab)
        lay, order = layout(); seen = {}
        for s_, box in lay.items():
            for tr in parked:
                if inside(box, *ctr(tr["avg"])): seen[s_] = min(seen.get(s_, 1e9), tr["anchor_vt"]); S["vtype"][s_] = tr["cls"]
        for s_ in lay:
            if s_ in seen: S["last_seen"][s_] = vt
            was = S["occ"].get(s_, False)
            now = (vt - S["last_seen"].get(s_, -99)) < OCC_GRACE
            if booked_now(s_): now = False           # booked slots are not processed
            if now and not was: S["since"][s_] = seen.get(s_, vt)
            if was and not now: new_charge(s_, S["last_seen"][s_] - S["since"].pop(s_, vt), S["vtype"].get(s_, "car"))
            S["occ"][s_] = now
        S["overlay"] = {"moving": [tr["box"] for tr in cur if tr not in parked],
                        "slots": [(lay[s_], "booked" if booked_now(s_) else "occ" if S["occ"].get(s_) else "free",
                                   f"R{i+1}-{j+1}") for i, row in enumerate(order) for j, s_ in enumerate(row)]}

def end_pass(vt, state):
    with LOCK:                                       # bill cars still parked, reset for the next loop
        for s_, was in list(S["occ"].items()):
            if was: new_charge(s_, max(0, vt - S["since"].pop(s_, vt)), S["vtype"].get(s_, "car"))
        S["occ"].clear(); S["since"].clear(); S["last_seen"].clear()
    state["tracks"].clear()
    if TRACKER:
        try: state["model"].predictor = None        # forget ByteTrack / BoT-SORT state for the next loop
        except Exception: pass

def placeholder(text):
    import numpy as np
    img = np.zeros((540, 960, 3), np.uint8); cv2.putText(img, text, (30, 270), 0, 0.9, (200, 200, 200), 2)
    return cv2.imencode(".jpg", img)[1].tobytes()

def player():
    """Plays the video (file at real speed, or live webcam/RTSP) and ALWAYS streams frames, however slow detection is."""
    import traceback
    S["jpg"] = placeholder("Loading video...")
    live = VIDEO_SOURCE is not None
    try:
        vids = [] if live else sorted(sum([glob.glob(f"{VIDEO_DIR}/*.{e}") for e in ("mp4", "avi", "mov", "mkv", "MP4", "MOV")], []))
        if not live and not vids:
            S["status"] = "No video found in ./videos"; S["jpg"] = placeholder("No video in ./videos folder"); return
        COL = {"free": (0, 200, 0), "occ": (60, 60, 255), "booked": (255, 160, 0)}
        offset = 0.0
        while True:                                  # loops forever (files) / reconnects (live sources)
            if live: cap = cv2.VideoCapture(int(VIDEO_SOURCE) if VIDEO_SOURCE.isdigit() else VIDEO_SOURCE); name = VIDEO_SOURCE
            else: cap = cv2.VideoCapture(vids[0]); name = os.path.basename(vids[0])
            fps = cap.get(cv2.CAP_PROP_FPS)
            if not fps or fps != fps or fps < 5 or fps > 120: fps = 25
            n, t0 = 0, time.time(); S["status"] = "live: " + name
            tick, shown = time.time(), 0
            while True:
                ok, frame = cap.read()
                if not ok: break
                n += 1; vt = offset + ((time.time() - t0) if live else n / fps); S["vt"] = vt
                if frame.shape[1] > 960:
                    f = 960 / frame.shape[1]; frame = cv2.resize(frame, None, fx=f, fy=f)
                SH["frame"], SH["vt"] = frame, vt; SH["seq"] += 1
                view = frame.copy(); ov = S.get("overlay") or {"moving": [], "slots": []}
                for x1, y1, x2, y2 in ov["moving"]: cv2.rectangle(view, (x1, y1), (x2, y2), (170, 170, 170), 1)
                for (x1, y1, x2, y2), st, lab in ov["slots"]:
                    cv2.rectangle(view, (x1, y1), (x2, y2), COL[st], 2)
                    cv2.putText(view, lab, (x1 + 4, y1 + 18), 0, 0.55, COL[st], 2)
                S["jpg"] = cv2.imencode(".jpg", view, [cv2.IMWRITE_JPEG_QUALITY, 75])[1].tobytes()
                shown += 1
                if time.time() - tick >= 1: S["fps"] = round(shown / (time.time() - tick), 1); shown, tick = 0, time.time()
                if not live:
                    wait = n / fps - (time.time() - t0)
                    if wait > 0: time.sleep(wait)
            cap.release()
            if n == 0:
                S["status"] = "cannot read video source"; S["jpg"] = placeholder("Cannot read video source"); time.sleep(3); continue
            if live:                                 # stream dropped: keep the timeline running and reconnect
                offset += time.time() - t0; S["status"] = "reconnecting..."; time.sleep(2); continue
            offset += n / fps; SH["pass_end"] = offset
    except Exception as e:
        traceback.print_exc(); S["status"] = "player crashed: " + repr(e)[:80]

def detector():
    import traceback
    try:
        from ultralytics import YOLO
        try:
            import torch; torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))
        except Exception: pass
        S["det"] = "loading " + YOLO_MODEL
        state = {"model": YOLO(YOLO_MODEL), "tracks": {}, "tid": 0, "ids": 0}
        S["det"] = "detecting"
    except Exception as e:
        traceback.print_exc(); S["det"] = "detector unavailable: " + repr(e)[:80]; return
    last = 0
    while True:
        if SH["pass_end"] is not None:
            v = SH["pass_end"]; SH["pass_end"] = None; end_pass(v, state); continue
        if SH["frame"] is None or SH["seq"] == last: time.sleep(0.02); continue
        last = SH["seq"]; t_ = time.time()
        try: process(SH["frame"], SH["vt"], state); S["det"] = "detecting"
        except Exception as e: traceback.print_exc(); S["det"] = "error: " + repr(e)[:80]
        S["det_ms"] = int((time.time() - t_) * 1000)
        time.sleep(max(0.05, 0.25 - (time.time() - t_)))   # leave CPU for the video stream

def logger_loop():
    """Every LOG_EVERY seconds store how many slots are free / occupied / booked (feeds the busy-hours heatmap)."""
    while True:
        time.sleep(LOG_EVERY)
        try:
            with LOCK: st = ["booked" if booked_now(s) else "occupied" if S["occ"].get(s) else "free" for s in S["slots"]]
            with db() as c:
                if st: c.execute("INSERT INTO occ_log VALUES(?,?,?,?)", (time.time(), st.count("free"), st.count("occupied"), st.count("booked")))
                c.execute("DELETE FROM occ_log WHERE ts<?", (time.time() - 30 * 86400,))
        except Exception as e: print("history log error:", e)

_started = False
def start_workers():
    global _started
    if _started: return
    _started = True
    for fn in (player, detector, logger_loop): threading.Thread(target=fn, daemon=True).start()

# ---------------- web ----------------
def refresh_session():
    """Re-read the role from the DB so a promoted/removed user is noticed, and make sure a CSRF token exists."""
    with db() as c: u = c.execute("SELECT role FROM users WHERE id=?", (session["uid"],)).fetchone()
    if not u: session.clear(); return False
    session["role"] = u["role"] or "user"; session.setdefault("csrf", secrets.token_hex(16)); return True

def need_login(f):
    def w(*a, **k):
        if "uid" not in session or (not request.path.startswith("/api") and not refresh_session()):
            return redirect("/auth") if not request.path.startswith("/api") else ("auth", 401)
        return f(*a, **k)
    w.__name__ = f.__name__; return w

def need_admin(f):
    def w(*a, **k):
        if "uid" not in session or not refresh_session(): return redirect("/auth") if not request.path.startswith("/api") else ("auth", 401)
        if session.get("role") != "admin": return ("admin only", 403)
        return f(*a, **k)
    w.__name__ = f.__name__; return w

@app.before_request
def csrf_guard():                                    # every state-changing /api call must carry the session's token
    if request.method in ("POST", "PUT", "PATCH", "DELETE") and request.path.startswith("/api/"):
        tok = session.get("csrf")
        if not tok or not hmac.compare_digest(request.headers.get("X-CSRF-Token", ""), tok): return jsonify(error="bad csrf token"), 403

FAILS = {}
def too_many(key, limit=5, window=300):
    now = time.time(); FAILS[key] = [t for t in FAILS.get(key, []) if now - t < window]
    return len(FAILS[key]) >= limit
def note_fail(key): FAILS.setdefault(key, []).append(time.time())
def jd():
    d = request.get_json(silent=True); return d if isinstance(d, dict) else {}

AUTHPAGE = """<!doctype html><meta name=viewport content="width=device-width,initial-scale=1"><title>Sign in - Smart Parking</title>
<style>*{box-sizing:border-box}body{margin:0;min-height:100vh;display:grid;place-items:center;font-family:Inter,system-ui,sans-serif;color:#e2e8f0;background:radial-gradient(circle at 0 0,#1e1b4b 0,transparent 40%),radial-gradient(circle at 100% 100%,#052e2b 0,transparent 40%),#0b1020}
.box{width:min(390px,92vw);background:#ffffff0d;border:1px solid #ffffff1a;border-radius:18px;padding:26px}
h1{margin:0 0 4px;font-size:22px}h1 span{color:#34d399}p.s{margin:0 0 16px;color:#94a3b8;font-size:13px}
.tabs{display:grid;grid-template-columns:repeat(3,1fr);gap:6px;margin-bottom:16px}.tabs button{background:#ffffff12;color:#cbd5e1;padding:9px 4px;border:0;border-radius:9px;font-weight:600;cursor:pointer;font-size:13px}
.tabs button.on{background:linear-gradient(135deg,#34d399,#06b6d4);color:#04210f}form{display:none;gap:10px}form.on{display:grid}
input{padding:11px;border-radius:9px;border:1px solid #ffffff1a;background:#0b1020;color:#fff;font-size:14px;outline:0}input:focus{border-color:#34d399}
.go{padding:12px;border:0;border-radius:9px;background:linear-gradient(135deg,#34d399,#06b6d4);color:#04210f;font-weight:700;cursor:pointer;font-size:14px}
.err{color:#fca5a5;font-size:13px;margin-bottom:12px}.ok{color:#6ee7b7;font-size:13px;margin-bottom:12px}small{color:#94a3b8}</style>
<div class=box><h1>&#128663; Smart<span>Parking</span></h1><p class=s>Find parking near you, see live slots and book.</p>
{% if first %}<div class=ok>No accounts yet - the first account you register becomes the admin.</div>{% endif %}
{% if err %}<div class=err>{{err}}</div>{% endif %}
<div class=tabs><button id=t-login onclick="sw('login')">Login</button><button id=t-register onclick="sw('register')">Register</button><button id=t-admin onclick="sw('admin')">Admin</button></div>
<form id=f-login method=post action=/auth/login><input name=email type=email placeholder=Email required><input name=pw type=password placeholder=Password required><button class=go>Login</button></form>
<form id=f-register method=post action=/auth/register><input name=name placeholder="Full name" required><input name=email type=email placeholder=Email required><input name=pw type=password minlength=6 placeholder="Password (min 6 characters)" required><button class=go>Create account</button></form>
<form id=f-admin method=post action=/auth/admin><input name=email type=email placeholder="Admin email" required><input name=pw type=password placeholder=Password required><button class=go>Admin login</button><small>Admins can see every booking and charge, and manage parkings.</small></form></div>
<script>function sw(t){for(const k of ['login','register','admin']){document.getElementById('f-'+k).classList.toggle('on',k==t);document.getElementById('t-'+k).classList.toggle('on',k==t)}}sw('{{tab}}')</script>"""

def render_auth(tab, err="", code=200):
    with db() as c: first = c.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0
    return render_template_string(AUTHPAGE, tab=tab, err=err, first=first), code

def start_session(u):
    session.clear()
    session["uid"], session["name"], session["role"], session["csrf"] = u["id"], u["name"], u["role"] or "user", secrets.token_hex(16)
    return redirect("/admin" if session["role"] == "admin" else "/nearby")

def try_login(tab, admin_only=False):
    key = "login:" + str(request.remote_addr)
    if too_many(key): return render_auth(tab, "Too many failed attempts. Wait a few minutes.", 429)
    with db() as c: u = c.execute("SELECT * FROM users WHERE email=?", (request.form.get("email", "").strip().lower(),)).fetchone()
    if u and check_password_hash(u["pw"], request.form.get("pw", "")) and (not admin_only or u["role"] == "admin"):
        FAILS.pop(key, None); return start_session(u)
    note_fail(key)
    return render_auth(tab, "Wrong email or password" + (", or this is not an admin account" if admin_only else ""), 401)

@app.route("/auth")
def auth_page():
    if "uid" in session: return redirect("/admin" if session.get("role") == "admin" else "/nearby")
    tab = request.args.get("tab", "login"); return render_auth(tab if tab in ("login", "register", "admin") else "login")

@app.route("/login")
def login_old(): return redirect("/auth?tab=login")
@app.route("/register")
def register_old(): return redirect("/auth?tab=register")

@app.route("/auth/login", methods=["POST"])
def auth_login(): return try_login("login")

@app.route("/auth/admin", methods=["POST"])
def auth_admin(): return try_login("admin", admin_only=True)

@app.route("/auth/register", methods=["POST"])
def auth_register():
    key = "reg:" + str(request.remote_addr); f = request.form
    name, email, pw = f.get("name", "").strip(), f.get("email", "").strip().lower(), f.get("pw", "")
    if too_many(key, 10, 3600): return render_auth("register", "Too many registrations from this address. Try again later.", 429)
    if not name or "@" not in email or len(pw) < 6: return render_auth("register", "Enter a name, a valid email and a password of at least 6 characters")
    note_fail(key)
    try:
        with db() as c:
            first = c.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0
            c.execute("INSERT INTO users(name,email,pw,role) VALUES(?,?,?,?)", (name, email, generate_password_hash(pw), "admin" if first else "user"))
            u = c.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
    except sqlite3.IntegrityError: return render_auth("register", "Email already registered")
    return start_session(u)                           # registered -> signed in -> straight to nearby parkings

@app.route("/logout")
def logout(): session.clear(); return redirect("/auth")

@app.route("/video_feed")
@need_login
def video_feed():
    def gen():
        last = None
        while True:
            j = S["jpg"]
            if j is not None and j is not last:
                last = j
                yield b"--f\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(j)).encode() + b"\r\n\r\n" + j + b"\r\n"
            time.sleep(0.03)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=f", headers={"Cache-Control": "no-cache"})

@app.route("/frame.jpg")
@need_login
def frame_jpg():
    return Response(S["jpg"] or b"", mimetype="image/jpeg", headers={"Cache-Control": "no-store"})

@app.route("/api/state")
@need_login
def state():
    adm = session.get("role") == "admin"
    with LOCK:
        rows = [[{"id": s, "label": f"R{i+1}-{j+1}",
                  "state": "booked" if booked_now(s) else "occupied" if S["occ"].get(s) else "free"}
                 for j, s in enumerate(r)] for i, r in enumerate(layout()[1])]
    status = (S["status"] + " | " + S.get("det", "")) if adm else ("online" if S["status"].startswith("live") else S["status"])
    out = {"rows": rows, "status": status, "vt": round(S["vt"], 1), "rate": RATE, "admin": adm, "pricing": pricing_text(), "auto": S["auto"]}
    with db() as c:
        if adm:
            out["charges"] = [dict(r) for r in c.execute("SELECT id,slot,seconds,amount,status,run,vtype FROM charges ORDER BY id DESC LIMIT 15")]
            out["revenue"] = c.execute("SELECT COALESCE(SUM(amount),0) FROM charges WHERE status='PAID'").fetchone()[0]
        else: out["charges"], out["revenue"] = [], None
        out["mine"] = [dict(r) for r in c.execute("SELECT id,slot,start,end,amount FROM bookings WHERE user_id=? AND end>? ORDER BY start", (session["uid"], time.time()))]
        if WALLET: out["balance"] = c.execute("SELECT balance FROM users WHERE id=?", (session["uid"],)).fetchone()[0] or 0
    return jsonify(out)

def slot_labels():
    with LOCK: return {s: f"R{i+1}-{j+1}" for i, r in enumerate(layout()[1]) for j, s in enumerate(r)}

def slot_counts():
    with LOCK: st = ["booked" if booked_now(s) else "occupied" if S["occ"].get(s) else "free" for s in S["slots"]]
    return {"free": st.count("free"), "total": len(st)}

def haversine(lat1, lon1, lat2, lon2):
    from math import radians, sin, cos, asin, sqrt
    a = sin(radians(lat2 - lat1) / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(radians(lon2 - lon1) / 2) ** 2
    return 6371 * 2 * asin(sqrt(a))

@app.route("/api/parkings")
@need_login
def parkings_list():
    try: lat, lng = float(request.args["lat"]), float(request.args["lng"]); have = -90 <= lat <= 90 and -180 <= lng <= 180
    except (KeyError, ValueError): have = False
    with db() as c: rows = [dict(r) for r in c.execute("SELECT * FROM parkings")]
    cnt = slot_counts()
    for r in rows:
        r["km"] = round(haversine(lat, lng, r["lat"], r["lng"]), 2) if have else None
        if r["live"]: r["free"], r["total"] = cnt["free"], cnt["total"]
    rows.sort(key=lambda r: (r["km"] is None, r["km"] or 0, r["name"]))
    return jsonify(parkings=rows, located=have)

def parking_fields(d):
    name, address = str(d.get("name", "")).strip()[:80], str(d.get("address", "")).strip()[:160]
    lat, lng = float(d["lat"]), float(d["lng"])
    if not name or not (-90 <= lat <= 90 and -180 <= lng <= 180): raise ValueError
    return name, address, lat, lng

@app.route("/api/parkings", methods=["POST"])
@need_admin
def parkings_add():
    try: f = parking_fields(jd())
    except (KeyError, ValueError, TypeError): return jsonify(error="Enter a name and a valid latitude / longitude"), 400
    with db() as c: pid = c.execute("INSERT INTO parkings(name,address,lat,lng,live) VALUES(?,?,?,?,0)", f).lastrowid
    return jsonify(ok=True, id=pid)

@app.route("/api/parkings/<int:pid>", methods=["PUT", "DELETE"])
@need_admin
def parkings_edit(pid):
    with db() as c:
        p = c.execute("SELECT * FROM parkings WHERE id=?", (pid,)).fetchone()
        if not p: return jsonify(error="not found"), 404
        if request.method == "DELETE":
            if p["live"]: return jsonify(error="The live-camera parking cannot be deleted"), 409
            c.execute("DELETE FROM parkings WHERE id=?", (pid,)); return jsonify(ok=True)
        try: f = parking_fields(jd())
        except (KeyError, ValueError, TypeError): return jsonify(error="Enter a name and a valid latitude / longitude"), 400
        c.execute("UPDATE parkings SET name=?, address=?, lat=?, lng=? WHERE id=?", (*f, pid))
    return jsonify(ok=True)

@app.route("/api/admin/bookings")
@need_admin
def admin_bookings():
    lab = slot_labels()
    with db() as c:
        rows = [dict(r) for r in c.execute("SELECT b.id,b.slot,b.start,b.end,b.amount,u.name,u.email FROM bookings b LEFT JOIN users u ON u.id=b.user_id ORDER BY b.start DESC LIMIT 200")]
    for r in rows: r["label"] = lab.get(r["slot"], f"#{r['slot']}")
    return jsonify(bookings=rows)

@app.route("/api/book", methods=["POST"])
@need_login
def book():
    d = jd()
    try: slot = int(d["slot"]); start_in = max(0, float(d.get("start_min", 0))) * 60; secs = float(d["seconds"])
    except (KeyError, ValueError, TypeError): return jsonify(error="invalid"), 400
    if secs <= 0 or secs > 86400 or slot not in S["slots"]: return jsonify(error="invalid"), 400
    s0 = time.time() + start_in; e0 = s0 + secs; amt = price(secs)
    with db() as c:
        if c.execute("SELECT 1 FROM bookings WHERE slot=? AND start<? AND end>?", (slot, e0, s0)).fetchone():
            return jsonify(error="Slot already booked for that time"), 409
        if start_in == 0 and S["occ"].get(slot): return jsonify(error="Slot is occupied"), 409
        if WALLET and c.execute("UPDATE users SET balance=balance-? WHERE id=? AND balance>=?", (amt, session["uid"], amt)).rowcount == 0:
            return jsonify(error="Not enough wallet balance - top up first"), 402
        bid = c.execute("INSERT INTO bookings(user_id,slot,start,end,amount) VALUES(?,?,?,?,?)", (session["uid"], slot, s0, e0, amt)).lastrowid
        u = c.execute("SELECT email FROM users WHERE id=?", (session["uid"],)).fetchone()
    if u: notify("Parking booking confirmed", f"Slot {slot}: {time.strftime('%d %b %H:%M', time.localtime(s0))} for {int(secs)} s, \u20b9{amt}.", u["email"])
    return jsonify(ok=True, charged=amt, id=bid)

def refund_amount(b):
    now = time.time()
    if now <= b["start"]: return b["amount"]
    if now >= b["end"]: return 0
    return int(b["amount"] * (b["end"] - now) / (b["end"] - b["start"]))

@app.route("/api/booking/<int:bid>/cancel", methods=["POST"])
@need_login
def cancel_booking(bid):
    with db() as c:
        b = c.execute("SELECT * FROM bookings WHERE id=?", (bid,)).fetchone()
        if not b or (b["user_id"] != session["uid"] and session.get("role") != "admin"): return jsonify(error="not found"), 404
        if b["end"] <= time.time(): return jsonify(error="Booking already finished"), 409
        ref = refund_amount(b) if WALLET else 0
        c.execute("DELETE FROM bookings WHERE id=?", (bid,))
        if ref: c.execute("UPDATE users SET balance=balance+? WHERE id=?", (ref, b["user_id"]))
    return jsonify(ok=True, refunded=ref)

@app.route("/api/booking/<int:bid>", methods=["PUT"])
@need_login
def edit_booking(bid):
    d = jd(); now = time.time()
    with db() as c:
        b = c.execute("SELECT * FROM bookings WHERE id=?", (bid,)).fetchone()
        if not b or (b["user_id"] != session["uid"] and session.get("role") != "admin"): return jsonify(error="not found"), 404
        if b["start"] <= now: return jsonify(error="Only bookings that have not started yet can be edited"), 409
        try:
            start_in = max(0, float(d["start_min"])) * 60 if "start_min" in d else b["start"] - now
            secs = float(d.get("seconds", b["end"] - b["start"]))
        except (ValueError, TypeError): return jsonify(error="invalid"), 400
        if secs <= 0 or secs > 86400: return jsonify(error="invalid"), 400
        s0 = now + start_in; e0 = s0 + secs
        if c.execute("SELECT 1 FROM bookings WHERE slot=? AND id<>? AND start<? AND end>?", (b["slot"], bid, e0, s0)).fetchone():
            return jsonify(error="Slot already booked for that time"), 409
        if start_in == 0 and S["occ"].get(b["slot"]): return jsonify(error="Slot is occupied"), 409
        amt = price(secs); diff = amt - b["amount"]
        if WALLET and diff > 0 and c.execute("UPDATE users SET balance=balance-? WHERE id=? AND balance>=?", (diff, b["user_id"], diff)).rowcount == 0:
            return jsonify(error="Not enough wallet balance - top up first"), 402
        if WALLET and diff < 0: c.execute("UPDATE users SET balance=balance-? WHERE id=?", (diff, b["user_id"]))   # negative diff = refund
        c.execute("UPDATE bookings SET start=?, end=?, amount=? WHERE id=?", (s0, e0, amt, bid))
    return jsonify(ok=True, amount=amt)

@app.route("/api/topup", methods=["POST"])
@need_login
def topup():
    if not WALLET: return jsonify(error="wallet is disabled (set WALLET=1)"), 404
    try: amt = float(jd().get("amount", 0))
    except (ValueError, TypeError): return jsonify(error="invalid"), 400
    if not 1 <= amt <= 10000: return jsonify(error="amount must be between 1 and 10000"), 400
    with db() as c:                                  # test mode: credit instantly (plug a real gateway in here)
        c.execute("UPDATE users SET balance=balance+? WHERE id=?", (amt, session["uid"]))
        bal = c.execute("SELECT balance FROM users WHERE id=?", (session["uid"],)).fetchone()[0]
    return jsonify(ok=True, balance=bal)

# ---- manual slot editor (admin) ----
def box_from(d):
    xs = sorted((float(d["x1"]), float(d["x2"]))); ys = sorted((float(d["y1"]), float(d["y2"])))
    if xs[1] - xs[0] < 10 or ys[1] - ys[0] < 10: raise ValueError
    return (xs[0], ys[0], xs[1], ys[1])

@app.route("/api/slots", methods=["POST"])
@need_admin
def slots_add():
    try: box = box_from(jd())
    except (KeyError, ValueError, TypeError): return jsonify(error="invalid box (at least 10 px each way)"), 400
    return jsonify(ok=True, id=add_slot(box))

@app.route("/api/slots/<int:sid>", methods=["PUT", "DELETE"])
@need_admin
def slots_edit(sid):
    if request.method == "DELETE": ok = drop_slot(sid)
    else:
        try: ok = move_slot(sid, box_from(jd()))
        except (KeyError, ValueError, TypeError): return jsonify(error="invalid box (at least 10 px each way)"), 400
    return (jsonify(ok=True), 200) if ok else (jsonify(error="no such slot"), 404)

@app.route("/api/slots/delete_at", methods=["POST"])
@need_admin
def slots_delete_at():
    d = jd()
    try: x, y = float(d["x"]), float(d["y"])
    except (KeyError, ValueError, TypeError): return jsonify(error="invalid"), 400
    with LOCK:
        hit = [((b[2] - b[0]) * (b[3] - b[1]), s) for s, b in layout()[0].items() if inside(b, x, y, 0)]
        if not hit: return jsonify(error="no slot there"), 404
        sid = min(hit)[1]; drop_slot(sid)
    return jsonify(ok=True, id=sid)

@app.route("/api/slots/clear", methods=["POST"])
@need_admin
def slots_clear():
    with LOCK:
        for s in list(S["slots"]): drop_slot(s)
    return jsonify(ok=True)

@app.route("/api/auto", methods=["POST"])
@need_admin
def auto_toggle():
    S["auto"] = bool(jd().get("on")); return jsonify(ok=True, on=S["auto"])

# ---- billing admin ----
@app.route("/api/retry/<int:cid>", methods=["POST"])
@need_admin
def retry(cid):
    with db() as c:
        ch = c.execute("SELECT * FROM charges WHERE id=?", (cid,)).fetchone()
        if not ch: return jsonify(error="not found"), 404
        if ch["status"] != "FAILED": return jsonify(error="only FAILED charges can be retried"), 409
        c.execute("UPDATE charges SET status='PENDING' WHERE id=?", (cid,))
    threading.Thread(target=tinyfish_pay, args=(cid, ch["token"], ch["amount"]), daemon=True).start()
    return jsonify(ok=True)

@app.route("/webhook/payment", methods=["POST"])     # payment provider -> us; send header X-Webhook-Secret and JSON {"id":..,"token":..}
def pay_webhook():
    if not WEBHOOK_SECRET or not hmac.compare_digest(request.headers.get("X-Webhook-Secret", ""), WEBHOOK_SECRET): return ("forbidden", 403)
    d = jd()
    try: settle(int(d.get("id", 0)), str(d.get("token", "")), "WEBHOOK")
    except (ValueError, TypeError): return jsonify(error="invalid"), 400
    return jsonify(ok=True)

def csv_safe(v):
    s = str(v if v is not None else "")
    return "'" + s if s[:1] in ("=", "+", "-", "@") else s

@app.route("/admin/export.csv")
@need_admin
def export_csv():
    out = io.StringIO(); w = csv.writer(out)
    with db() as c:
        if request.args.get("kind") == "bookings":
            lab = slot_labels(); w.writerow(["id", "user", "email", "slot", "from", "to", "amount"])
            fmt = lambda t: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t))
            for r in c.execute("SELECT b.id,b.slot,b.start,b.end,b.amount,u.name,u.email FROM bookings b LEFT JOIN users u ON u.id=b.user_id ORDER BY b.start DESC"):
                w.writerow([r["id"], csv_safe(r["name"]), csv_safe(r["email"]), lab.get(r["slot"], r["slot"]), fmt(r["start"]), fmt(r["end"]), r["amount"]])
            name = "bookings.csv"
        elif request.args.get("kind") == "daily":
            w.writerow(["date", "charges", "seconds", "revenue_paid"])
            for r in c.execute("SELECT date(created,'unixepoch','localtime') d, COUNT(*), SUM(seconds), SUM(CASE WHEN status='PAID' THEN amount ELSE 0 END) FROM charges GROUP BY d ORDER BY d DESC"): w.writerow(list(r))
            name = "revenue_by_day.csv"
        else:
            w.writerow(["id", "slot", "vehicle", "seconds", "amount", "status", "created", "run"])
            for r in c.execute("SELECT id,slot,vtype,seconds,amount,status,created,run FROM charges ORDER BY id"):
                w.writerow([r["id"], r["slot"], csv_safe(r["vtype"]), r["seconds"], r["amount"], r["status"],
                            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r["created"])), csv_safe(r["run"])])
            name = "charges.csv"
    return Response(out.getvalue(), mimetype="text/csv", headers={"Content-Disposition": f"attachment; filename={name}"})

@app.route("/api/history")
@need_login
def history():
    with db() as c:
        rows = c.execute("SELECT strftime('%H', ts,'unixepoch','localtime') h, AVG(100.0*(occupied+booked)/(free+occupied+booked)) p "
                         "FROM occ_log WHERE free+occupied+booked>0 GROUP BY h").fetchall()
        avg = c.execute("SELECT AVG(seconds), COUNT(*) FROM charges").fetchone()
    hours = [None] * 24
    for r in rows: hours[int(r["h"])] = round(r["p"], 1)
    known = [(v, i) for i, v in enumerate(hours) if v is not None]
    return jsonify(hours=hours, avg_stay=round(avg[0] or 0), stays=avg[1], busiest=max(known)[1] if known else None)

@app.route("/health")
def health():
    ok = S["status"].startswith("live") and S.get("det") == "detecting"
    return jsonify(ok=ok, video=S["status"], detector=S.get("det", "starting"), stream_fps=S["fps"], detector_ms=S["det_ms"],
                   slots=len(S["slots"]), uptime_s=int(time.time() - S["started"]), model=YOLO_MODEL, tracker=TRACKER or "iou"), (200 if ok else 503)

@app.route("/pay/<int:cid>", methods=["GET", "POST"])   # fake payment page TinyFish operates (one-time token)
def pay(cid):
    ch = db().execute("SELECT * FROM charges WHERE id=?", (cid,)).fetchone()
    tok = request.values.get("t")
    if not ch or tok != ch["token"]: return "Invalid link", 403
    if request.method == "POST": settle(cid, tok); ch = db().execute("SELECT * FROM charges WHERE id=?", (cid,)).fetchone()
    return render_template_string("""<!doctype html><title>Pay</title><body style="font-family:system-ui;max-width:360px;margin:40px auto">
    <h2>Parking payment #{{c.id}}</h2><p>Slot {{c.slot}} &middot; {{c.seconds}} s &middot; <b>Amount: {{c.amount}} rupees</b></p>
    {% if c.status=='PAID' %}<h3 id=ok>Payment successful</h3>{% else %}
    <form method=post action="/pay/{{c.id}}?t={{t}}"><input name=card placeholder="Card number" required><br><br>
    <input name=exp placeholder="MM/YY" required> <input name=cvv placeholder="CVV" required><br><br>
    <button id=pay-btn>Pay Now</button></form>{% endif %}""", c=ch, t=tok)

DASH = """<!doctype html><meta name=viewport content="width=device-width,initial-scale=1"><title>{{ptitle}} - Smart Parking</title>
<style>*{box-sizing:border-box}:root{--bg:#0b1020;--card:#ffffff0d;--bd:#ffffff1a;--mut:#94a3b8}
body{margin:0;font-family:Inter,system-ui,sans-serif;color:#e2e8f0;background:radial-gradient(circle at 0 0,#1e1b4b 0,transparent 35%),radial-gradient(circle at 100% 100%,#052e2b 0,transparent 35%),var(--bg);min-height:100vh}
header{display:flex;justify-content:space-between;align-items:center;padding:14px 24px;border-bottom:1px solid var(--bd);backdrop-filter:blur(10px);position:sticky;top:0;z-index:5;background:#0b1020cc}
.logo{font-weight:800;font-size:18px}.logo span{color:#34d399}.live{display:inline-flex;gap:6px;align-items:center;font-size:13px;color:var(--mut)}
.dot{width:8px;height:8px;border-radius:50%;background:#34d399;animation:p 1.4s infinite}@keyframes p{50%{opacity:.25}}
.user{font-size:14px;color:var(--mut)}.user a{color:#67e8f9;text-decoration:none;margin-left:8px}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;padding:16px 24px 0}
.stat{background:var(--card);border:1px solid var(--bd);border-radius:14px;padding:14px 16px}.stat small{color:var(--mut)}.stat b{display:block;font-size:26px;margin-top:2px}
main{display:grid;grid-template-columns:1.1fr 1fr;gap:16px;padding:16px 24px 24px}
@media(max-width:950px){main{grid-template-columns:1fr}.stats{grid-template-columns:repeat(2,1fr)}}
.card{background:var(--card);border:1px solid var(--bd);border-radius:16px;padding:16px}.card h3{margin:0 0 12px;font-size:15px;color:#cbd5e1;font-weight:600}
img{width:100%;border-radius:12px;display:block;background:#000}.info{margin-top:10px;font-size:13px;color:var(--mut)}
.leg{display:flex;gap:14px;font-size:12px;color:var(--mut);margin-bottom:12px}.leg i{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:5px}
.row{display:grid;gap:10px;margin-bottom:10px}.slot{border-radius:14px;padding:12px 8px;text-align:center;border:1px solid;transition:.2s;min-height:104px;display:flex;flex-direction:column;justify-content:center;gap:4px}
.slot b{font-size:13px}.slot .ic{font-size:26px}.slot small{font-size:11px;text-transform:uppercase;letter-spacing:.08em}
.free{background:#10b98118;border-color:#10b98177;color:#6ee7b7}.occupied{background:#ef444418;border-color:#ef444477;color:#fca5a5}.booked{background:#f59e0b18;border-color:#f59e0b77;color:#fcd34d}
.free:hover{transform:translateY(-2px);box-shadow:0 8px 24px #10b98133}
button{padding:7px 10px;border:0;border-radius:8px;background:linear-gradient(135deg,#34d399,#06b6d4);color:#04210f;font-weight:700;cursor:pointer;font-size:12px}
table{width:100%;font-size:13px;border-collapse:collapse}th{color:var(--mut);font-weight:500;text-align:left;padding:6px}td{padding:7px 6px;border-top:1px solid var(--bd)}
.pill{padding:2px 9px;border-radius:99px;font-size:11px;font-weight:700}.PAID,.SIMULATED{background:#10b98126;color:#6ee7b7}.PENDING{background:#f59e0b26;color:#fcd34d}.FAILED{background:#ef444426;color:#fca5a5}
#m{position:fixed;inset:0;background:#000b;backdrop-filter:blur(4px);display:none;place-items:center;z-index:9}
#m>div{background:#111831;border:1px solid var(--bd);padding:24px;border-radius:18px;width:340px;display:grid;gap:10px;box-shadow:0 30px 80px #000}
#m h3{margin:0}label{font-size:12px;color:var(--mut);display:grid;gap:4px}
input{padding:10px;border-radius:9px;border:1px solid var(--bd);background:#0b1020;color:#fff;font-size:14px;outline:0}input:focus{border-color:#34d399}
.cost{font-size:22px;font-weight:800;color:#6ee7b7}.g{background:#334155;color:#e2e8f0}#er{color:#fca5a5;font-size:13px}
#t{position:fixed;bottom:20px;right:20px;background:#10b981;color:#04210f;padding:12px 18px;border-radius:12px;font-weight:700;display:none;z-index:10}
</style>
<header><a class=logo href=/nearby style=text-decoration:none;color:inherit>&#128663; Smart<span>Parking</span></a><div class=live><i class=dot></i><span id=st>connecting</span></div>
<div class=user>{{name}}{% if is_admin %} (admin){% endif %}<span id=bal></span><a href=/nearby>Nearby parkings</a><a href=/bookings>My bookings</a>{% if is_admin %}<a href=/admin>Admin</a>{% endif %}{% if admin %}<a href="/admin/export.csv?kind=bookings">Bookings CSV</a><a href="/admin/export.csv?kind=charges">Charges CSV</a><a href="/admin/export.csv?kind=daily">Daily CSV</a>{% endif %}<a href=/logout>Logout</a></div></header>
<div class=stats><div class=stat><small>Free</small><b id=sf style=color:#6ee7b7>0</b></div><div class=stat><small>Occupied</small><b id=so style=color:#fca5a5>0</b></div>
<div class=stat><small>Booked</small><b id=sb style=color:#fcd34d>0</b></div><div class=stat><small>Occupancy</small><b id=sp>-</b></div>{% if admin %}<div class=stat><small>Revenue collected</small><b id=sr>&#8377;0</b></div>{% endif %}</div>
<main><div class=card><h3>{{ptitle}} &middot; live camera</h3><div style="position:relative"><img id=cam alt="Live camera">{% if admin %}<div id=ov style="position:absolute;inset:0;display:none;cursor:crosshair"><div id=rb style="position:absolute;border:2px dashed #34d399;display:none;pointer-events:none"></div></div>{% endif %}</div><div class=info id=info></div>{% if admin %}<div style="display:flex;gap:8px;margin-top:10px;flex-wrap:wrap"><button id=tg onclick=tog()>Edit slots</button><button class=g onclick=autoT()>Auto-detect: <span id=au>on</span></button><button class=g onclick=clr()>Clear all slots</button></div><div class=info>Edit mode: drag on the video to add a slot, click a slot to delete it (to resize, delete and redraw).</div>{% endif %}</div>
<div><div class=card><h3>Parking grid</h3><div class=leg><span><i style=background:#10b981></i>Free</span><span><i style=background:#ef4444></i>Occupied</span><span><i style=background:#f59e0b></i>Booked</span></div>
<div id=grid class=info>Waiting for parked cars to be detected...</div></div>
{% if admin %}<div class=card style=margin-top:16px><h3>All bookings</h3><div style="overflow-x:auto"><table id=ab></table></div></div>
<div class=card style=margin-top:16px><h3>Charges &middot; auto-paid by TinyFish agent</h3><table id=ch></table></div>
<div class=card style=margin-top:16px><h3>Parkings</h3><div style="overflow-x:auto"><table id=pk></table></div>
<div style="display:grid;gap:8px;margin-top:12px"><input id=pn placeholder="Name"><input id=pa placeholder="Address (optional)">
<div style="display:flex;gap:8px"><input id=pla placeholder=Latitude style="width:50%"><input id=plo placeholder=Longitude style="width:50%"></div>
<div style="display:flex;gap:8px;flex-wrap:wrap"><button onclick=pHere()>Use my location</button><button onclick=pSave()><span id=psl>Add parking</span></button><button class=g onclick=pClear()>Clear</button></div></div></div>{% endif %}<div class=card style=margin-top:16px>
<h3 style=margin-top:16px>My bookings</h3><div id=mine class=info></div></div>
<div class=card style=margin-top:16px><h3>Busy hours (heatmap)</h3><div id=hm style="display:grid;grid-template-columns:repeat(24,1fr);gap:3px"></div><div class=info id=hmi></div></div></div></main>
<div id=m><div><h3 id=mt></h3><label>Start in (minutes, 0 = now)<input id=sm type=number value=0 min=0></label>
<label>Duration (seconds)<input id=ds type=number value=60 min=1></label><div class=cost id=cost></div>
<label>Card (demo)<input value="4242 4242 4242 4242"></label><div style="display:flex;gap:8px"><input value="12/30" style="width:50%"><input value=123 style="width:50%"></div>
<button style="padding:12px;font-size:14px" onclick=pay()>Pay &amp; book</button><button class=g style="padding:10px;font-size:13px" onclick="m.style.display='none'">Cancel</button><div id=er></div></div></div>
<div id=t></div>
<script>let slot=null,rate=10;const H={'Content-Type':'application/json','X-CSRF-Token':'{{csrf}}'};const $=i=>document.getElementById(i),m=$('m');
function open_(id,l){slot=id;$('mt').textContent='Book slot '+l;m.style.display='grid';$('er').textContent='';up()}
function up(){$('cost').textContent='Estimated total: \u20B9'+(+$('ds').value*rate)}$('ds').oninput=up;
function toast(x){const t=$('t');t.textContent=x;t.style.display='block';setTimeout(()=>t.style.display='none',3000)}
async function pay(){const r=await fetch('/api/book',{method:'POST',headers:H,body:JSON.stringify({slot,start_min:$('sm').value,seconds:$('ds').value})});
const j=await r.json();if(r.ok){m.style.display='none';toast('Booked! Charged \u20B9'+j.charged);tick()}else $('er').textContent=j.error||'failed'}
async function tick(){const s=await (await fetch('/api/state')).json();rate=s.rate;
$('st').textContent=s.status;$('info').textContent='Video time '+s.vt+'s \u00B7 '+s.pricing;
const all=s.rows.flat(),n=k=>all.filter(c=>c.state==k).length;$('sf').textContent=n('free');$('so').textContent=n('occupied');$('sb').textContent=n('booked');if($('sr'))$('sr').textContent='\u20B9'+(s.revenue||0);$('sp').textContent=all.length?Math.round(100*(n('occupied')+n('booked'))/all.length)+'%':'-';if($('au'))$('au').textContent=s.auto?'on':'off';
const cols=Math.max(1,...s.rows.map(r=>r.length));
$('grid').className='';$('grid').innerHTML=s.rows.map(r=>`<div class=row style="grid-template-columns:repeat(${cols},1fr)">`+r.map(c=>`<div class="slot ${c.state}"><span class=ic>${c.state=='free'?'🟢':c.state=='booked'?'🕒':'🚗'}</span><b>${c.label}</b><small>${c.state}</small>${c.state=='free'?`<button onclick="open_(${c.id},'${c.label}')">Book slot</button>`:''}</div>`).join('')+'</div>').join('')||'Waiting for parked cars to be detected...';
if($('ch'))$('ch').innerHTML='<tr><th>#<th>Slot<th>Type<th>Sec<th>Amount<th>Status<th></tr>'+s.charges.map(c=>`<tr><td>${c.id}<td>${c.slot}<td>${c.vtype||'car'}<td>${c.seconds}<td>\u20B9${c.amount}<td><span class="pill ${c.status}">${c.status}</span><td><a href="/receipt/${c.id}" target=_blank style=color:#67e8f9>receipt</a>${c.status=='FAILED'?` <button onclick="retry(${c.id})">Retry</button>`:''}</tr>`).join('');
$('mine').innerHTML=s.mine.map(b=>`Slot ${b.slot}: ${new Date(b.start*1e3).toLocaleTimeString()} - ${new Date(b.end*1e3).toLocaleTimeString()} (\u20B9${b.amount}) <button onclick="editB(${b.id})">Edit</button> <button class=g onclick="cancelB(${b.id})">Cancel</button>`).join('<br>')||'No upcoming bookings';
if(s.balance!==undefined)$('bal').innerHTML=' &middot; Wallet \u20B9'+Math.round(s.balance)+' <button onclick="topup()">Top up</button>'}
async function api(u,body,method){const r=await fetch(u,{method:method||'POST',headers:H,body:JSON.stringify(body||{})});const j=await r.json().catch(()=>({}));if(!r.ok)toast(j.error||'failed');return j}
async function cancelB(id){if(!confirm('Cancel this booking?'))return;const j=await api('/api/booking/'+id+'/cancel');if(j.ok){toast(j.refunded?'Cancelled, refunded \u20B9'+j.refunded:'Booking cancelled');tick();loadAdmin()}}
async function editB(id){const sm=prompt('Start in how many minutes? (0 = now)','0');if(sm===null)return;const d=prompt('Duration in seconds?','60');if(d===null)return;const j=await api('/api/booking/'+id,{start_min:sm,seconds:d},'PUT');if(j.ok){toast('Booking updated');tick()}}
async function retry(id){const j=await api('/api/retry/'+id);if(j.ok)toast('Retrying payment...');tick()}
async function topup(){const a=prompt('Top-up amount (test mode)','500');if(!a)return;const j=await api('/api/topup',{amount:a});if(j.ok){toast('Wallet topped up');tick()}}
async function heat(){const h=await (await fetch('/api/history')).json();$('hm').innerHTML=h.hours.map((v,i)=>`<div title="${i}:00 - ${v==null?'no data':v+'% busy'}" style="height:34px;border-radius:4px;background:${v==null?'#ffffff10':'rgba(52,211,153,'+(0.15+v/130)+')'}"></div>`).join('');$('hmi').textContent='Hours 0-23 (local time). Average stay: '+h.avg_stay+'s over '+h.stays+' charges'+(h.busiest!=null?'. Busiest hour: '+h.busiest+':00':'')}
const esc=s=>String(s==null?'':s).replace(/[&<>"']/g,c=>'&#'+c.charCodeAt(0)+';');
let PK=[],pid_=null;
async function loadAdmin(){if(!$('ab'))return;
const b=await (await fetch('/api/admin/bookings')).json(),now=Date.now()/1e3;
$('ab').innerHTML='<tr><th>#<th>User<th>Slot<th>From<th>To<th>Amount<th>Status<th></tr>'+(b.bookings.map(x=>{const st=x.end<now?'finished':x.start<=now?'active':'upcoming';
return `<tr><td>${x.id}<td>${esc(x.name)}<br><small>${esc(x.email)}</small><td>${esc(x.label)}<td>${new Date(x.start*1e3).toLocaleString()}<td>${new Date(x.end*1e3).toLocaleString()}<td>\u20B9${x.amount}<td>${st}<td>${st=='finished'?'':`<button class=g onclick="cancelB(${x.id})">Cancel</button>`}</tr>`}).join('')||'<tr><td colspan=8>No bookings yet</tr>');
const p=await (await fetch('/api/parkings')).json();PK=p.parkings;
$('pk').innerHTML='<tr><th>Name<th>Location<th></tr>'+PK.map((q,i)=>`<tr><td>${esc(q.name)}${q.live?' <span class="pill PAID">live</span>':''}<td>${q.lat.toFixed(4)}, ${q.lng.toFixed(4)}<td><button class=g onclick="pEdit(${i})">Edit</button>${q.live?'':` <button class=g onclick="pDel(${q.id})">Delete</button>`}</tr>`).join('')}
function pEdit(i){const p=PK[i];pid_=p.id;$('pn').value=p.name;$('pa').value=p.address||'';$('pla').value=p.lat;$('plo').value=p.lng;$('psl').textContent='Save changes'}
function pClear(){pid_=null;['pn','pa','pla','plo'].forEach(i=>$(i).value='');$('psl').textContent='Add parking'}
function pHere(){if(!navigator.geolocation)return toast('Location unavailable');navigator.geolocation.getCurrentPosition(p=>{$('pla').value=p.coords.latitude.toFixed(6);$('plo').value=p.coords.longitude.toFixed(6)},()=>toast('Location unavailable'))}
async function pSave(){const body={name:$('pn').value,address:$('pa').value,lat:$('pla').value,lng:$('plo').value};const j=pid_?await api('/api/parkings/'+pid_,body,'PUT'):await api('/api/parkings',body);if(j.ok){toast('Saved');pClear();loadAdmin()}}
async function pDel(id){if(confirm('Delete this parking?')){const j=await api('/api/parkings/'+id,{},'DELETE');if(j.ok)loadAdmin()}}
let edit=false;
function tog(){edit=!edit;$('ov').style.display=edit?'block':'none';$('tg').textContent=edit?'Done editing':'Edit slots'}
async function autoT(){const j=await api('/api/auto',{on:$('au').textContent!='on'});$('au').textContent=j.on?'on':'off'}
async function clr(){if(confirm('Delete all slots?')){await api('/api/slots/clear');tick()}}
if($('ov')){let a=null;const ov=$('ov'),rb=$('rb'),im=$('cam');
const fp=e=>{const r=im.getBoundingClientRect(),k=im.naturalWidth/r.width;return[(e.clientX-r.left)*k,(e.clientY-r.top)*k,e.clientX-r.left,e.clientY-r.top]};
ov.onmousedown=e=>{a=fp(e);rb.style.display='block';rb.style.left=a[2]+'px';rb.style.top=a[3]+'px';rb.style.width='0px';rb.style.height='0px'};
ov.onmousemove=e=>{if(!a)return;const b=fp(e);rb.style.left=Math.min(a[2],b[2])+'px';rb.style.top=Math.min(a[3],b[3])+'px';rb.style.width=Math.abs(a[2]-b[2])+'px';rb.style.height=Math.abs(a[3]-b[3])+'px'};
ov.onmouseup=async e=>{if(!a)return;const b=fp(e),s=a;a=null;rb.style.display='none';
if(Math.abs(b[0]-s[0])<8&&Math.abs(b[1]-s[1])<8)await api('/api/slots/delete_at',{x:b[0],y:b[1]});else await api('/api/slots',{x1:s[0],y1:s[1],x2:b[0],y2:b[1]});tick()}}
const cam=$('cam');function nf(){cam.onload=()=>setTimeout(nf,40);cam.onerror=()=>setTimeout(nf,500);cam.src='/frame.jpg?'+Date.now()}nf();
setInterval(tick,1000);tick();heat();setInterval(heat,60000);loadAdmin();setInterval(loadAdmin,5000)</script>"""

RECEIPT = """<!doctype html><meta name=viewport content="width=device-width,initial-scale=1"><title>Receipt</title>
<body style="font-family:system-ui;max-width:420px;margin:40px auto;padding:0 16px;color:#111"><h2>Smart Parking &middot; {{title}}</h2>
<table style="width:100%;border-collapse:collapse">{% for k, v in lines %}<tr style="border-top:1px solid #ddd"><td style="padding:8px 0;color:#555">{{k}}</td><td style="text-align:right"><b>{{v}}</b></td></tr>{% endfor %}</table>
<p style="color:#777;font-size:13px">Thank you for parking with us.</p><button onclick="print()">Print</button> <a href=/>Back</a></body>"""

BOOKINGS = """<!doctype html><meta name=viewport content="width=device-width,initial-scale=1"><title>My bookings</title>
<body style="font-family:system-ui;max-width:760px;margin:30px auto;padding:0 16px;color:#e2e8f0;background:#0b1020"><h2>My bookings</h2>
<p><a href=/nearby style=color:#67e8f9>&larr; Nearby parkings</a></p><table style="width:100%;border-collapse:collapse">
<tr><th align=left>#<th align=left>Slot<th align=left>From<th align=left>To<th align=left>Amount<th></tr>
{% for b in rows %}<tr style="border-top:1px solid #ffffff22"><td>{{b.id}}<td>{{labels.get(b.slot, b.slot)}}<td><span class=t data-t="{{b.start}}"></span><td><span class=t data-t="{{b.end}}"></span>
<td>&#8377;{{b.amount}}<td><a href="/receipt/b/{{b.id}}" style=color:#67e8f9>receipt</a>{% if b.end > now %} <a href=# onclick="cx({{b.id}});return false" style=color:#fca5a5>cancel</a>{% endif %}</tr>
{% else %}<tr><td colspan=6>No bookings yet</td></tr>{% endfor %}</table>
<script>document.querySelectorAll('.t').forEach(e=>e.textContent=new Date(e.dataset.t*1e3).toLocaleString());
async function cx(id){if(!confirm('Cancel this booking?'))return;const r=await fetch('/api/booking/'+id+'/cancel',{method:'POST',headers:{'X-CSRF-Token':'{{csrf}}'}});
if(r.ok)location.reload();else alert((await r.json()).error||'failed')}</script></body>"""

@app.route("/bookings")
@need_login
def bookings_page():
    with db() as c: rows = c.execute("SELECT * FROM bookings WHERE user_id=? ORDER BY start DESC", (session["uid"],)).fetchall()
    return render_template_string(BOOKINGS, rows=rows, now=time.time(), csrf=session.get("csrf", ""), labels=slot_labels())

@app.route("/receipt/<int:cid>")
@need_admin
def receipt(cid):
    with db() as c: ch = c.execute("SELECT * FROM charges WHERE id=?", (cid,)).fetchone()
    if not ch: return "Not found", 404
    lines = [("Receipt #", ch["id"]), ("Slot", ch["slot"]), ("Vehicle", ch["vtype"] or "car"), ("Duration", f"{ch['seconds']} s"),
             ("Amount", f"\u20b9{ch['amount']}"), ("Status", ch["status"]), ("Date", time.strftime("%d %b %Y %H:%M", time.localtime(ch["created"])))]
    return render_template_string(RECEIPT, title=f"Payment receipt #{cid}", lines=lines)

@app.route("/receipt/b/<int:bid>")
@need_login
def booking_receipt(bid):
    with db() as c: b = c.execute("SELECT * FROM bookings WHERE id=?", (bid,)).fetchone()
    if not b or (b["user_id"] != session["uid"] and session.get("role") != "admin"): return "Not found", 404
    fmt = lambda t: time.strftime("%d %b %Y %H:%M:%S", time.localtime(t))
    lines = [("Booking #", b["id"]), ("Slot", b["slot"]), ("From", fmt(b["start"])), ("To", fmt(b["end"])), ("Amount", f"\u20b9{b['amount']}")]
    return render_template_string(RECEIPT, title=f"Booking receipt #{bid}", lines=lines)

THEME = """*{box-sizing:border-box}body{margin:0;font-family:Inter,system-ui,sans-serif;color:#e2e8f0;background:radial-gradient(circle at 0 0,#1e1b4b 0,transparent 35%),radial-gradient(circle at 100% 100%,#052e2b 0,transparent 35%),#0b1020;min-height:100vh}
header{display:flex;justify-content:space-between;align-items:center;padding:14px 24px;border-bottom:1px solid #ffffff1a;background:#0b1020cc;position:sticky;top:0}
.logo{font-weight:800;font-size:18px;color:#e2e8f0;text-decoration:none}.logo span{color:#34d399}.user{font-size:14px;color:#94a3b8}.user a{color:#67e8f9;text-decoration:none;margin-left:12px}
.info{font-size:13px;color:#94a3b8;margin:6px 0}"""

NEARBY = """<!doctype html><meta name=viewport content="width=device-width,initial-scale=1"><title>Nearby parkings - Smart Parking</title>
<style>""" + THEME + """
.pk{display:flex;justify-content:space-between;gap:12px;text-decoration:none;color:inherit;background:#ffffff0d;border:1px solid #ffffff1a;border-radius:14px;padding:14px 16px;margin-top:12px}.pk:hover{border-color:#34d399}
.tag{background:#34d39922;color:#6ee7b7;border-radius:99px;padding:2px 8px;font-size:11px;font-weight:700}.sub{color:#94a3b8;font-size:12px;margin-top:3px}.r{text-align:right}.live2{color:#6ee7b7}
button{padding:9px 14px;border:0;border-radius:8px;background:linear-gradient(135deg,#34d399,#06b6d4);color:#04210f;font-weight:700;cursor:pointer}</style>
<header><a class=logo href=/nearby>&#128663; Smart<span>Parking</span></a><div class=user>{{name}}{% if is_admin %} (admin){% endif %}<a href=/bookings>My bookings</a>{% if is_admin %}<a href=/admin>Admin</a>{% endif %}<a href=/logout>Logout</a></div></header>
<main style="max-width:760px;margin:0 auto;padding:20px 16px"><h2 style=margin:0>Parkings near you</h2><div class=info id=loc>Finding your location...</div>
<div id=list></div><button id=rl onclick=go() style="display:none;margin-top:14px">Use my location</button></main>
<script>const $=i=>document.getElementById(i),esc=s=>String(s==null?'':s).replace(/[&<>"']/g,c=>'&#'+c.charCodeAt(0)+';');
async function load(lat,lng){const j=await (await fetch('/api/parkings'+(lat==null?'':'?lat='+lat+'&lng='+lng))).json();
$('list').innerHTML=j.parkings.map((p,i)=>`<a class=pk href="/parking/${p.id}"><div><b>${esc(p.name)}</b>${i==0&&j.located?' <span class=tag>Nearest</span>':''}<div class=sub>${esc(p.address||'')}</div></div><div class=r>${p.km!=null?'<b>'+p.km+' km</b>':''}<div class=sub>${p.live?`<span class=live2>&#9679; Live video</span> &middot; ${p.free} of ${p.total} slots free`:'No live camera (demo)'}</div></div></a>`).join('')||'<div class=info>No parkings yet.</div>'}
function go(){$('rl').style.display='none';if(!navigator.geolocation){$('loc').textContent='Location is not supported by this browser - showing all parkings.';load();return}
$('loc').textContent='Finding your location...';
navigator.geolocation.getCurrentPosition(p=>{$('loc').textContent='Sorted by distance from your current location.';load(p.coords.latitude,p.coords.longitude)},
()=>{$('loc').textContent='Could not get your location (allow location access; browsers need https or localhost) - showing all parkings.';$('rl').style.display='inline-block';load()},{timeout:10000,maximumAge:60000})}
go();setInterval(()=>{if($('list').children.length)go()},60000)</script>"""

NOLIVE = """<!doctype html><meta name=viewport content="width=device-width,initial-scale=1"><title>{{p.name}}</title>
<style>""" + THEME + """</style><header><a class=logo href=/nearby>&#128663; Smart<span>Parking</span></a><div class=user><a href=/nearby>Nearby parkings</a><a href=/logout>Logout</a></div></header>
<main style="max-width:560px;margin:40px auto;padding:0 16px"><h2>{{p.name}}</h2><p class=info>{{p.address}}</p>
<p>This parking has no live camera yet, so slots can't be shown or booked here.</p><p><a href=/nearby style=color:#67e8f9>&larr; Back to nearby parkings</a></p></main>"""

@app.route("/")
def home(): return redirect("/nearby" if "uid" in session else "/auth")

@app.route("/nearby")
@need_login
def nearby():
    return render_template_string(NEARBY, name=session["name"], is_admin=session.get("role") == "admin")

@app.route("/parking/<int:pid>")
@need_login
def parking_page(pid):
    with db() as c: p = c.execute("SELECT * FROM parkings WHERE id=?", (pid,)).fetchone()
    if not p: return "Parking not found", 404
    if not p["live"]: return render_template_string(NOLIVE, p=p)
    return render_template_string(DASH, name=session["name"], csrf=session["csrf"], admin=False, is_admin=session.get("role") == "admin", ptitle=p["name"])

@app.route("/admin")
@need_admin
def admin_page():
    return render_template_string(DASH, name=session["name"], csrf=session["csrf"], admin=True, is_admin=True, ptitle="Admin")

if os.getenv("AUTOSTART") == "1": start_workers()     # for gunicorn: AUTOSTART=1 gunicorn -w 1 --threads 8 app:app

if __name__ == "__main__":
    start_workers()
    app.run(host="0.0.0.0", port=5000, threaded=True)
