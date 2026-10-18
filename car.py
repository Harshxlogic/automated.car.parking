"""Smart Parking - single-file Flask app (video -> YOLO -> slots -> billing -> dashboard).

Run:     python app.py        (dev)
         AUTOSTART=1 gunicorn -w 1 --threads 8 -b 0.0.0.0:5000 app:app     (production; keep exactly ONE worker)
Config:  a .env file (KEY=value lines), environment variables, or an optional config.json; real env vars win over both files.
  Pricing    RATE, FREE_SECS, MIN_CHARGE, DAILY_CAP (per slot), PEAK_START, PEAK_END, PEAK_MULT, BUS_MULT, TRUCK_MULT
  Detection  PARK_SECS, OCC_GRACE, CONF, YOLO_MODEL, TRACKER (bytetrack | botsort | empty = built-in IoU tracker)
  Video      VIDEO_SOURCE (webcam index, rtsp:// or http:// URL; empty = first file in the lot's folder; lot1 only)
  Lots       one folder per parking lot: ./videos/lot1, ./videos/lot2 ... (loose files in ./videos still count as lot1)
             DUAL_LOTS=lot2 (comma list) = lots whose video has TWO cameras side by side: left = gate/plate camera, right = overhead colour camera
  Two-camera DUAL_FILL, DUAL_DIFF, DUAL_BASE_SECS (empty-lot reference; or put videos/lot2/empty.jpg), DUAL_DEBUG=1 (show fill %), MIN_READS, OCR_SCALE, ENTRY_TTL
             needs: pip install easyocr   (or pytesseract + the tesseract program)
  Guard      GUARD_PHONE (10 digits or +91...), TWILIO_SID, TWILIO_TOKEN, TWILIO_FROM = the guard gets a voice call on wrong parking; GUARD_EMAIL, WRONG_SECS
  Behaviour  AUTO_SLOTS=0 (no automatic slot discovery), WALLET=0 (switch the prepaid wallet off), START_BALANCE (rupees for new accounts, default 1), LOG_EVERY (history interval, s)
  Security   SECRET_KEY, PUBLIC_URL (https => secure cookies), BEHIND_PROXY=1, WEBHOOK_SECRET
  Payments   RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET (Razorpay is on once both are set), RAZORPAY_WEBHOOK_SECRET (optional, for /webhook/razorpay),
             AUTO_TINYFISH=0 (unpaid charges wait for wallet / Razorpay instead of the TinyFish agent), FREE_TOPUP=1 (free test top-ups even with Razorpay),
             TEST_EMAIL, TEST_PASSWORD, TEST_BALANCE (ready-made test driver, created once an admin exists)
  Email      SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, SMTP_FROM, ADMIN_EMAIL
The first account that registers becomes the admin.
"""
import os, glob, time, sqlite3, secrets, threading, csv, io, json, hmac, smtplib, re, random
from contextlib import contextmanager
from email.message import EmailMessage
import cv2, requests
from xml.sax.saxutils import escape as xml_escape
from flask import Flask, request, session, redirect, jsonify, Response, render_template_string
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.middleware.proxy_fix import ProxyFix

os.chdir(os.path.dirname(os.path.abspath(__file__)))   # videos/, .env, parking.db ... are found next to this file, whatever folder you start it from

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
WALLET = os.getenv("WALLET", "1") != "0"          # prepaid wallet is ON unless WALLET=0
START_BALANCE = env_f("START_BALANCE", 1)       # rupees every new account starts with
CLAIM_TTL = 12 * 3600                            # a "bill my wallet" claim on a slot lasts at most 12 h
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
DISCOVER_URL = os.getenv("DISCOVER_URL", "https://www.openstreetmap.org/")   # site TinyFish searches for nearby parkings
RZP_KEY, RZP_SECRET = (os.getenv("RAZORPAY_KEY_ID") or "").strip(), (os.getenv("RAZORPAY_KEY_SECRET") or "").strip()
RZP_WEBHOOK = (os.getenv("RAZORPAY_WEBHOOK_SECRET") or "").strip()
REMIND_SMS = os.getenv("REMIND_SMS", "1") != "0"           # reminders also go out as SMS when TWILIO_SID / TWILIO_TOKEN / TWILIO_FROM are set
REMIND_MINS = env_f("REMIND_MINS", 0)                     # >0: e-mail a driver once when a charge has been pending this many minutes (needs SMTP_HOST)
RZP = bool(RZP_KEY and RZP_SECRET)                    # Razorpay is on as soon as both keys are in .env
AUTO_TINYFISH = os.getenv("AUTO_TINYFISH", "1") != "0"  # 1 = unpaid charges go to the TinyFish agent (as before); 0 = they wait for wallet / Razorpay
FREE_TOPUP = os.getenv("FREE_TOPUP") == "1"           # free test top-ups; only matters when Razorpay is configured (without keys top-up stays free test credit)
TEST_EMAIL, TEST_PW, TEST_BALANCE = os.getenv("TEST_EMAIL", "test@parking.com").strip().lower(), os.getenv("TEST_PASSWORD", "test1234"), env_f("TEST_BALANCE", 500)
DB = "parking.db"
# Parking plans: pay once, get some parked time at a discount (extra time is charged at the normal rate). Override with PLANS_JSON in .env, e.g.
# [{"key":"week","name":"Week Pack","days":7,"secs":300,"off":0.3}]   secs = parked seconds included (the same video-seconds clock billing uses), off = discount 0-0.9
PLANS = [{"key": "day", "name": "Day Pass", "days": 1, "secs": 60, "off": 0.20},
         {"key": "week", "name": "Week Pack", "days": 7, "secs": 300, "off": 0.30},
         {"key": "month", "name": "Month Pack", "days": 30, "secs": 1200, "off": 0.40}]
try:
    if os.getenv("PLANS_JSON"): PLANS = [{"key": re.sub(r"[^a-z0-9_]", "", str(p["key"]).lower()), "name": str(p["name"]), "days": max(1, int(p["days"])), "secs": max(1, int(p["secs"])), "off": min(0.9, max(0.0, float(p["off"])))} for p in json.loads(os.environ["PLANS_JSON"])]
except Exception as _e: print("PLANS_JSON ignored:", _e)

EXTS = ("mp4", "avi", "mov", "mkv", "webm", "m4v", "MP4", "MOV", "AVI", "MKV")
def lot_dirs():
    """{lot key: folder}. Every folder in ./videos named lot1, Lot 2, LOT_3, lot-4 ... is one parking lot (lot1 always exists)."""
    out = {}
    for d in glob.glob(os.path.join(VIDEO_DIR, "*")):
        m = os.path.isdir(d) and re.fullmatch(r"lot[\s_-]*(\d+)", os.path.basename(d).strip(), re.I)
        if m: out.setdefault(f"lot{int(m.group(1))}", d)
    out.setdefault("lot1", os.path.join(VIDEO_DIR, "lot1"))
    return dict(sorted(out.items(), key=lambda kv: int(kv[0][3:])))
LOT_DIRS = lot_dirs(); LOT_KEYS = list(LOT_DIRS); FIRST = LOT_KEYS[0]
DUAL = {k.strip() for k in os.getenv("DUAL_LOTS", "lot2").split(",") if k.strip()}   # two-camera lots (plate cam + overhead colour cam)
def lot_videos(lot):
    v = sorted(sum([glob.glob(os.path.join(LOT_DIRS[lot], f"*.{e}")) for e in EXTS], []))
    if not v and lot == FIRST: v = sorted(sum([glob.glob(f"{VIDEO_DIR}/*.{e}") for e in EXTS], []))   # old layout: ./videos/*.mp4
    return v

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
# An older parking.db (from an earlier version of this project) can hold tables with other columns. Such a table is renamed
# to <name>_old_<time> (data kept, never read again) and a fresh one is created, so the app never crashes on a stale file.
_BASE = {"users": "id,name,email,pw", "bookings": "id,user_id,slot,start,end,amount", "charges": "id,slot,seconds,amount,status,token,run,created",
         "slots": "id,x1,y1,x2,y2", "occ_log": "ts,free,occupied,booked", "parkings": "id,name,address,lat,lng,live", "claims": "id,user_id,slot,created",
         "vehicles": "id,plate,owner,phone,email,colour,fake,user_id", "parked": "slot,lot,plate,since",
         "alerts": "id,ts,lot,slot,label,plate,expected_plate,booking_id,booked_by,status"}
with db() as _c:
    for _t, _cols in _BASE.items():
        _have = {_r[1] for _r in _c.execute(f"PRAGMA table_info({_t})")}
        if _have and not set(_cols.split(",")) <= _have:
            _c.execute(f"ALTER TABLE {_t} RENAME TO {_t}_old_{int(time.time())}")
            print(f"parking.db: table '{_t}' had an older layout - kept as '{_t}_old_*', created a new one")
with db() as c:
    c.executescript("""
    CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY, name TEXT, email TEXT UNIQUE, pw TEXT);
    CREATE TABLE IF NOT EXISTS bookings(id INTEGER PRIMARY KEY, user_id INT, slot INT, start REAL, end REAL, amount INT);
    CREATE TABLE IF NOT EXISTS charges(id INTEGER PRIMARY KEY, slot INT, seconds INT, amount INT,
        status TEXT, token TEXT, run TEXT, created REAL);
    CREATE TABLE IF NOT EXISTS slots(id INTEGER PRIMARY KEY, x1 INT, y1 INT, x2 INT, y2 INT);
    CREATE TABLE IF NOT EXISTS occ_log(ts REAL, free INT, occupied INT, booked INT);
    CREATE TABLE IF NOT EXISTS parkings(id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, address TEXT, lat REAL, lng REAL, live INT DEFAULT 0);
    CREATE TABLE IF NOT EXISTS claims(id INTEGER PRIMARY KEY, user_id INT, slot INT, created REAL);
    CREATE TABLE IF NOT EXISTS vehicles(id INTEGER PRIMARY KEY, plate TEXT UNIQUE, owner TEXT, phone TEXT, email TEXT, colour TEXT, fake INT DEFAULT 1, user_id INT);
    CREATE TABLE IF NOT EXISTS parked(slot INTEGER PRIMARY KEY, lot TEXT, plate TEXT UNIQUE, since REAL);
    CREATE TABLE IF NOT EXISTS alerts(id INTEGER PRIMARY KEY, ts REAL, lot TEXT, slot INT, label TEXT, plate TEXT, expected_plate TEXT, booking_id INT, booked_by TEXT, status TEXT);
    CREATE TABLE IF NOT EXISTS rzp_orders(order_id TEXT PRIMARY KEY, kind TEXT, charge_id INT, user_id INT, amount REAL, status TEXT, payment_id TEXT, created REAL);
    CREATE TABLE IF NOT EXISTS visits(id INTEGER PRIMARY KEY, lot TEXT, slot INT, plate TEXT, vtype TEXT, entered REAL, left REAL, seconds REAL, loop_no INT, boot INT, charge_id INT, user_id INT,
        status TEXT, source TEXT, covered INT DEFAULT 0);
    CREATE INDEX IF NOT EXISTS visits_plate ON visits(plate);
    CREATE TABLE IF NOT EXISTS user_plans(id INTEGER PRIMARY KEY, user_id INT, plan TEXT, name TEXT, start REAL, end REAL, included REAL, used REAL DEFAULT 0, price REAL, order_id TEXT, status TEXT DEFAULT 'ACTIVE');
    CREATE TABLE IF NOT EXISTS usage_model(user_id INTEGER PRIMARY KEY, trained REAL, sig TEXT, data TEXT);
    CREATE TABLE IF NOT EXISTS live_snap(lot TEXT, slot INT, plate TEXT, vtype TEXT, seconds REAL, updated REAL, PRIMARY KEY(lot, slot));
    CREATE TABLE IF NOT EXISTS wallet_tx(id INTEGER PRIMARY KEY, ts REAL, user_id INT, kind TEXT, amount REAL, ref TEXT, balance REAL);""")
    for _sql in ("ALTER TABLE users ADD COLUMN role TEXT DEFAULT 'user'", "ALTER TABLE users ADD COLUMN balance REAL DEFAULT 0",
                 "ALTER TABLE charges ADD COLUMN vtype TEXT DEFAULT 'car'", "ALTER TABLE charges ADD COLUMN user_id INT",
                 "ALTER TABLE charges ADD COLUMN booking_id INT", "ALTER TABLE parkings ADD COLUMN rate REAL",
                 "ALTER TABLE parkings ADD COLUMN status TEXT DEFAULT 'active'", "ALTER TABLE parkings ADD COLUMN lot TEXT",
                 "ALTER TABLE users ADD COLUMN phone TEXT", "ALTER TABLE users ADD COLUMN plate TEXT", "ALTER TABLE charges ADD COLUMN plate TEXT",
                 "ALTER TABLE slots ADD COLUMN lot TEXT DEFAULT 'lot1'", "ALTER TABLE occ_log ADD COLUMN lot TEXT DEFAULT 'lot1'",
                 "ALTER TABLE charges ADD COLUMN method TEXT", "ALTER TABLE charges ADD COLUMN paid_at REAL", "ALTER TABLE charges ADD COLUMN lot TEXT", "ALTER TABLE charges ADD COLUMN reminded REAL",
                 "ALTER TABLE charges ADD COLUMN plan_secs REAL DEFAULT 0", "ALTER TABLE rzp_orders ADD COLUMN plan TEXT"):
        try: c.execute(_sql)
        except sqlite3.OperationalError: pass          # column already there
    c.execute("UPDATE charges SET paid_at=created WHERE status='PAID' AND paid_at IS NULL")
    if not c.execute("SELECT 1 FROM visits").fetchone():       # first start with the visit log: keep the history that only exists as charges
        c.execute("INSERT INTO visits(lot,slot,plate,vtype,entered,left,seconds,loop_no,boot,charge_id,user_id,status,source,covered) SELECT COALESCE(lot,'lot1'),slot,plate,COALESCE(vtype,'car'),created-seconds,created,seconds,0,0,id,user_id,'CLOSED','imported',0 FROM charges WHERE booking_id IS NULL")      # older rows: paid time unknown, use the charge time
    if not c.execute("SELECT 1 FROM users WHERE role='admin'").fetchone():     # upgrade: oldest account becomes admin
        c.execute("UPDATE users SET role='admin' WHERE id=(SELECT MIN(id) FROM users)")
    # Live parking = USICT, GGSIPU campus, Sector 16C Dwarka (override with LIVE_LAT / LIVE_LNG, or edit it in the admin page)
    _lat, _lng = env_f("LIVE_LAT", 28.595016), env_f("LIVE_LNG", 77.018942)
    _LIVE_NAME, _LIVE_ADDR = os.getenv("LIVE_NAME", "USICT Dwarka (live camera)"), os.getenv("LIVE_ADDRESS", "USICT, GGSIPU, Sector 16C, Dwarka, New Delhi")
    _DEMOS = (("City Centre Parking (demo)", 0.009, 0.006, 0.8), ("Station Road Parking (demo)", -0.014, 0.011, 1.5), ("Mall Parking (demo)", 0.021, -0.017, 1.2))
    if not c.execute("SELECT 1 FROM parkings").fetchone():     # first run: the parking that has your video + a few demo ones
        c.execute("INSERT INTO parkings(name,address,lat,lng,live) VALUES(?,?,?,?,1)", (_LIVE_NAME, _LIVE_ADDR, _lat, _lng))
        for _n, _dl, _dg, _m in _DEMOS:
            c.execute("INSERT INTO parkings(name,address,lat,lng,live,rate) VALUES(?,?,?,?,0,?)",
                      (_n, "Demo entry - no live camera", _lat + _dl, _lng + _dg, round(RATE * _m, 2)))
    else:                                                     # upgrade an existing database that still has the old default location
        _lv = c.execute("SELECT * FROM parkings WHERE live=1").fetchone()
        if _lv and abs(_lv["lat"] - 28.6692) < 1e-4 and abs(_lv["lng"] - 77.4538) < 1e-4:
            _dl, _dg = _lat - _lv["lat"], _lng - _lv["lng"]
            c.execute("UPDATE parkings SET lat=lat+?, lng=lng+? WHERE live=0 AND address LIKE 'Demo entry%'", (_dl, _dg))
            c.execute("UPDATE parkings SET lat=?, lng=?, name=CASE WHEN name='Smart Parking (live camera)' THEN ? ELSE name END, "
                      "address=CASE WHEN COALESCE(address,'')='' THEN ? ELSE address END WHERE live=1", (_lat, _lng, _LIVE_NAME, _LIVE_ADDR))
        for _n, _dl, _dg, _m in _DEMOS:
            c.execute("UPDATE parkings SET rate=? WHERE name=? AND rate IS NULL", (round(RATE * _m, 2), _n))
    for _i, _k in enumerate(LOT_KEYS):                       # one live parking per ./videos/lotN folder
        if c.execute("SELECT 1 FROM parkings WHERE lot=?", (_k,)).fetchone(): continue
        _old = c.execute("SELECT id FROM parkings WHERE live=1 AND lot IS NULL ORDER BY id LIMIT 1").fetchone() if _i == 0 else None
        if _old: c.execute("UPDATE parkings SET lot=? WHERE id=?", (_k, _old["id"]))
        else: c.execute("INSERT INTO parkings(name,address,lat,lng,live,lot) VALUES(?,?,?,?,1,?)",
                        (f"Smart Parking Lot {_i + 1} (live camera)", "Gate camera + overhead camera (dual feed)" if _k in DUAL else "Live camera", _lat + 0.0006 * _i, _lng + 0.0004 * _i, _k))

# ---------------- vision / tracking ----------------
def new_state(lot):
    return {"slots": {}, "occ": {}, "since": {}, "last_seen": {}, "vtype": {}, "who": {}, "vt": 0.0, "status": "starting", "jpg": None,
            "auto": AUTO_SLOTS and lot not in DUAL, "fps": 0, "det_ms": 0, "started": time.time(), "alpha": lot in DUAL}
# One state + one player->detector hand-off per lot. S and SH below always mean "the lot THIS thread works on":
# request threads get it from ?lot= (see pick_lot), worker threads set it once, other code can use `with using(lot):`.
LOTS = {k: {"S": new_state(k), "SH": {"frame": None, "vt": 0.0, "seq": 0, "pass_end": None}} for k in LOT_KEYS}
_tl = threading.local()
def _cur(): return getattr(_tl, "lot", None) or FIRST
class _Cur:
    def __init__(self, k): self.k = k
    def _d(self): return LOTS[_cur()][self.k]
    def __getitem__(self, i): return self._d()[i]
    def __setitem__(self, i, v): self._d()[i] = v
    def get(self, i, d=None): return self._d().get(i, d)
S, SH = _Cur("S"), _Cur("SH")
@contextmanager
def using(lot):
    prev = getattr(_tl, "lot", None); _tl.lot = lot
    try: yield
    finally: _tl.lot = prev
LOCK = threading.RLock()
X = {}                                             # lot-independent messages (e.g. TinyFish discovery status)
_NID = [0]                                         # slot ids are unique across ALL lots, so bookings / charges stay valid

# ---- fake vehicle identities (plate, owner, phone) kept in the database ----
FAKE_FIRST = ["Aarav", "Vihaan", "Ishaan", "Kabir", "Rohan", "Arjun", "Dev", "Karan", "Neha", "Priya", "Ananya", "Diya", "Meera", "Riya", "Sana", "Tara", "Vikram", "Aditya", "Nikhil", "Pooja"]
FAKE_LAST = ["Sharma", "Verma", "Gupta", "Singh", "Mehta", "Khanna", "Reddy", "Nair", "Iyer", "Joshi", "Malik", "Bansal", "Kapoor", "Chopra", "Yadav", "Rao"]
COLOURS = ["Red", "Blue", "Green", "Yellow", "White", "Black", "Orange"]
def seed_vehicles(n=40):
    """First run: n fake cars (plate, owner, phone, colour) that lots without a plate reader hand out to parked cars."""
    with db() as c:
        if c.execute("SELECT 1 FROM vehicles WHERE fake=1").fetchone(): return
        r = random.Random(7); plates = set()
        while len(plates) < n:
            plates.add(f"{r.choice(['DL', 'UP', 'HR', 'RJ'])}{r.randint(1, 16)}{''.join(r.choice('ABCDEFGHKLMNPRSTUVXYZ') for _ in range(r.choice((2, 3))))}{r.randint(1000, 9999)}")
        for plate in sorted(plates):
            nm = f"{r.choice(FAKE_FIRST)} {r.choice(FAKE_LAST)}"
            try: c.execute("INSERT INTO vehicles(plate,owner,phone,email,colour,fake) VALUES(?,?,?,?,?,1)",
                           (plate, nm, f"90000{r.randint(0, 99999):05d}", nm.lower().replace(" ", ".") + "@example.com", r.choice(COLOURS)))
            except sqlite3.IntegrityError: pass
seed_vehicles()
with db() as _c: _c.execute("DELETE FROM parked")      # nobody is parked right after a restart

def vehicle_info(plates):
    plates = [p for p in plates if p]
    if not plates: return {}
    with db() as c:
        return {r["plate"]: dict(r) for r in c.execute(f"SELECT plate,owner,phone,colour FROM vehicles WHERE plate IN ({','.join('?' * len(plates))})", plates)}

def who_view(v, adm):
    """Admins see everything about the car in a slot; normal users only see the owner's name."""
    if not v: return None
    return {"name": v["owner"], "plate": v["plate"], "phone": v["phone"], "colour": v["colour"]} if adm else {"name": v["owner"]}

def assign_vehicle(slot):
    """Lots without a plate reader (lot1): the car that just parked here gets a fake identity from the pool.
    Deterministic per slot, so a slot shows the same fake car on every loop of the video."""
    with db() as c:
        pool = [r["plate"] for r in c.execute("SELECT plate FROM vehicles WHERE fake=1 ORDER BY id")]
        taken = {r["plate"] for r in c.execute("SELECT plate FROM parked")}
        for k in range(len(pool)):
            plate = pool[(slot + k) % len(pool)]
            if plate not in taken:
                c.execute("INSERT OR REPLACE INTO parked(slot,lot,plate,since) VALUES(?,?,?,?)", (slot, _cur(), plate, time.time()))
                S["who"][slot] = plate; return

def release_vehicle(slot):
    S["who"].pop(slot, None)
    with db() as c: c.execute("DELETE FROM parked WHERE slot=?", (slot,))

def release_all():
    for s_ in list(S["who"]): release_vehicle(s_)

def slot_busy(slot): return any(L["S"]["occ"].get(slot) for L in LOTS.values())

with db() as _c:                                   # slots survive restarts
    for _r in _c.execute("SELECT * FROM slots"):
        _k = _r["lot"] if _r["lot"] in LOTS else FIRST
        LOTS[_k]["S"]["slots"][_r["id"]] = (_r["x1"], _r["y1"], _r["x2"], _r["y2"]); _NID[0] = max(_NID[0], _r["id"])

def add_slot(box):
    with LOCK:
        _NID[0] += 1; sid = _NID[0]; b = tuple(int(v) for v in box); S["slots"][sid] = b
        with db() as c: c.execute("INSERT OR REPLACE INTO slots(id,x1,y1,x2,y2,lot) VALUES(?,?,?,?,?,?)", (sid, *b, _cur()))
    return sid

def seed_dual_slots():
    """Default 12-slot layout (rows A and B, 6 each) for the supplied two-camera video, in the 960-px-wide frame the player uses.
    Only used while the lot has no slots; for other footage use the admin slot editor."""
    xs, k = [731, 809, 887, 965, 1043, 1121, 1199], 0.75
    for y1, y2 in ((122, 206), (322, 406)):
        for a, b in zip(xs, xs[1:]): add_slot((a * k, y1 * k, b * k, y2 * k))
for _k in DUAL & set(LOTS):
    with using(_k):
        if not S["slots"]: seed_dual_slots()

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
            _sec = max(0, S["last_seen"].get(sid, S["vt"]) - S["since"].get(sid, S["vt"])); new_charge(sid, _sec, S["vtype"].get(sid, "car")); visit_out(sid, _sec, "slot_removed")
        S["slots"].pop(sid)
        for k in ("occ", "since", "last_seen", "vtype", "who"): S[k].pop(sid, None)
        with db() as c: c.execute("DELETE FROM slots WHERE id=?", (sid,)); c.execute("DELETE FROM parked WHERE slot=?", (sid,))
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
# (SH = the player -> detector hand-off of the current lot, defined with LOTS above)

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

def slot_label(i, j):          # lot1: R1-1, R1-2 ...   two-camera lots: A1, A2 ... B6 (matches the markings on the overhead view)
    return f"{chr(65 + i)}{j + 1}" if S.get("alpha") else f"R{i+1}-{j+1}"

def inside(b, x, y, pad=0.1):
    pw, ph = (b[2]-b[0]) * pad, (b[3]-b[1]) * pad
    return b[0]-pw <= x <= b[2]+pw and b[1]-ph <= y <= b[3]+ph

def booked_now(slot):
    t = time.time()
    with db() as c:
        return c.execute("SELECT 1 FROM bookings WHERE slot=? AND start<=? AND end>?", (slot, t, t)).fetchone() is not None

def slot_state(s):
    """free / occupied / booked. A two-camera lot can see a car inside a booked slot, so there the parked car wins (lot1 is unchanged)."""
    if _cur() in DUAL and S["occ"].get(s): return "occupied"
    return "booked" if booked_now(s) else "occupied" if S["occ"].get(s) else "free"

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

def claim_owner(slot):
    """The driver who pressed 'Bill my wallet' on this slot (if any, and not older than CLAIM_TTL)."""
    with db() as c:
        r = c.execute("SELECT user_id FROM claims WHERE slot=? AND created>? ORDER BY id DESC LIMIT 1", (slot, time.time() - CLAIM_TTL)).fetchone()
    return r["user_id"] if r else None

BOOT = int(time.time())                                    # one id per app start, so visits from different runs / video loops stay apart

def open_visit(slot):
    """A car has just parked in this slot: write the IN row (plate, lot, slot, time). Rows are never deleted or merged, so every replay of the video adds new visits."""
    lot, now, vt = _cur(), time.time(), S["vt"]
    ago = min(max(0.0, vt - S["since"].get(slot, vt)), 3600.0)
    try:
        with db() as c:
            c.execute("UPDATE visits SET status='CLOSED', left=?, source='unclosed' WHERE lot=? AND slot=? AND status='OPEN'", (now, lot, slot))
            c.execute("INSERT INTO visits(lot,slot,plate,vtype,entered,loop_no,boot,status) VALUES(?,?,?,?,?,?,?,'OPEN')", (lot, slot, S["who"].get(slot), S["vtype"].get(slot, "car"), now - ago, S.get("loops", 0) + 1, BOOT))
    except Exception as e: print("visit log:", e)

def visit_out(slot, seconds, source="exit", covered=False):
    """The car leaves (or the video loops / the app restarts): close the visit with the OUT time and parked seconds and link its charge. Then the owner's learning model is refreshed."""
    lot, now, plate = _cur(), time.time(), S["who"].get(slot)
    uid = plate_owner(plate)
    try:
        with db() as c:
            ch = c.execute("SELECT id,user_id FROM charges WHERE lot=? AND slot=? AND created>=? AND id NOT IN (SELECT charge_id FROM visits WHERE charge_id IS NOT NULL) ORDER BY id DESC LIMIT 1", (lot, slot, now - 5)).fetchone()
            cid = ch["id"] if ch else None; uid = uid or (ch["user_id"] if ch else None)
            v = c.execute("SELECT id FROM visits WHERE lot=? AND slot=? AND status='OPEN' ORDER BY id DESC LIMIT 1", (lot, slot)).fetchone()
            if v: c.execute("UPDATE visits SET plate=COALESCE(?,plate), left=?, seconds=?, status='CLOSED', source=?, charge_id=?, user_id=?, covered=? WHERE id=?", (plate, now, seconds, source, cid, uid, int(bool(covered)), v["id"]))
            else: c.execute("INSERT INTO visits(lot,slot,plate,vtype,entered,left,seconds,loop_no,boot,charge_id,user_id,status,source,covered) VALUES(?,?,?,?,?,?,?,?,?,?,?,'CLOSED',?,?)",
                            (lot, slot, plate, S["vtype"].get(slot, "car"), now - seconds, now, seconds, S.get("loops", 0) + 1, BOOT, cid, uid, source, int(bool(covered))))
        if uid: get_model(uid)
    except Exception as e: print("visit log:", e)

def new_charge(slot, seconds, vtype="car"):
    _bill_stay(slot, seconds, vtype)

def _bill_stay(slot, seconds, vtype="car"):
    """Bill a finished stay. If a driver claimed the slot with 'Bill my wallet', deduct it from their wallet right here
    (one atomic UPDATE, no web agent needed). If there is no claim or the wallet is too low, pay through the gateway."""
    amt = price(seconds, vtype, slot)
    if amt <= 0: return
    tok, paid = secrets.token_urlsafe(12), False
    plate = S["who"].get(slot)
    uid = (claim_owner(slot) if WALLET else None) or plate_owner(plate)    # a "Bill my wallet" claim first, else the wallet of the car's registered owner
    cov = 0.0
    if uid:                                                # an active parking plan pays for parked time first
        cov = use_plan_secs(uid, seconds)
        if cov > 0:
            amt = price(seconds - cov, vtype, slot)
            if amt <= 0:
                with db() as c: c.execute("INSERT INTO charges(slot,seconds,amount,status,token,created,vtype,user_id,plate,lot,method,paid_at,run,plan_secs) VALUES(?,?,0,'PAID',?,?,?,?,?,?,'PLAN',?,'covered by plan',?)",
                                          (slot, int(round(seconds)), tok, time.time(), vtype, uid, plate, _cur(), time.time(), cov))
                return
    with db() as c:
        cid = c.execute("INSERT INTO charges(slot,seconds,amount,status,token,created,vtype,user_id,plate,lot,plan_secs) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                        (slot, int(round(seconds)), amt, "PENDING", tok, time.time(), vtype, uid, plate, _cur(), cov)).lastrowid
        if uid and WALLET:
            if c.execute("UPDATE users SET balance=balance-? WHERE id=? AND balance>=?", (amt, uid, amt)).rowcount:
                c.execute("UPDATE charges SET status='PAID', run='WALLET', method='WALLET', paid_at=? WHERE id=?", (time.time(), cid)); paid = True; wtx(c, uid, "CHARGE", -amt, f"charge #{cid}")
            else: c.execute("UPDATE charges SET run=? WHERE id=?", ("wallet too low - using gateway" if AUTO_TINYFISH else "wallet too low - waiting for payment", cid))
            c.execute("DELETE FROM claims WHERE slot=?", (slot,))
    if paid: paid_alert(cid); return
    if AUTO_TINYFISH: threading.Thread(target=tinyfish_pay, args=(cid, tok, amt), daemon=True).start()

def tf_run(url, goal, timeout):
    return requests.post("https://agent.tinyfish.ai/v1/automation/run", timeout=timeout,
        headers={"X-API-Key": TF_KEY, "Content-Type": "application/json"}, json={"url": url, "goal": goal}).json()

def finish(cid, ok, run):
    """Final state of a gateway payment. A failed payment for a booking releases the slot again."""
    with db() as c:
        n = c.execute("UPDATE charges SET status=?, run=?, paid_at=?, method=CASE WHEN ? THEN COALESCE(method,'TINYFISH') ELSE method END WHERE id=? AND status='PENDING'",
                      ("PAID" if ok else "FAILED", run, time.time() if ok else None, 1 if ok else 0, cid)).rowcount
        ch = c.execute("SELECT booking_id FROM charges WHERE id=?", (cid,)).fetchone()
        if n and not ok and ch and ch["booking_id"]: c.execute("DELETE FROM bookings WHERE id=?", (ch["booking_id"],))
    if n and ok: paid_alert(cid)

def tinyfish_pay(cid, tok, amt):
    """Use a TinyFish web agent to pay the charge on a test payment page, with the payer's details filled in."""
    if not TF_KEY:  # fallback simulation
        time.sleep(2); settle(cid, tok, "SIMULATED"); return
    with db() as c:
        u = c.execute("SELECT u.name, u.email FROM charges ch JOIN users u ON u.id=ch.user_id WHERE ch.id=?", (cid,)).fetchone()
    nm, em = (u["name"], u["email"]) if u else ("Test User", "test@example.com")
    if PAYMENT_URL:   # agent pays on an existing PUBLIC test payment page
        goal = (f"This is a TEST-MODE payment page (no real money). If it asks for an amount, enter {amt}. "
                f"Fill any required fields with: name {nm}, email {em}, phone 9999999999. "
                f"Pay with test card 4242 4242 4242 4242, expiry 12/30, CVV 123 (if OTP/3D-secure appears use the test success option). "
                f"Submit the payment and wait for a success confirmation. "
                f"Return JSON {{\"paid\": true}} only if success is shown, else {{\"paid\": false}}.")
        try:
            r = tf_run(PAYMENT_URL, goal, 600)
            res = r.get("result") or {}
            ok = r.get("status") == "COMPLETED" and isinstance(res, dict) and res.get("paid") is True
            finish(cid, ok, str(r.get("run_id") or r.get("status")))
        except Exception as e: finish(cid, False, str(e)[:80])
        return
    goal = (f"This is a demo payment page. The form is already filled in; check the card number is 4242 4242 4242 4242, expiry 12/30, CVV 123, "
            f"confirm the amount is {amt} rupees, click the 'Pay Now' button, wait for the 'Payment successful' message. "
            f"Return JSON {{\"paid\": true}} if you see it, else {{\"paid\": false}}.")
    try:
        r = tf_run(f"{PUBLIC_URL}/pay/{cid}?t={tok}", goal, 300)
        run = str(r.get("run_id") or r.get("status"))
        with db() as c:
            paid = c.execute("SELECT status FROM charges WHERE id=?", (cid,)).fetchone()["status"] == "PAID"
            if paid: c.execute("UPDATE charges SET run=? WHERE id=?", (run, cid))
        if not paid: finish(cid, False, run)
    except Exception as e: finish(cid, False, str(e)[:80])

def settle(cid, tok, run=None):
    with db() as c:
        n = c.execute("UPDATE charges SET status='PAID', run=COALESCE(?,run), method=COALESCE(method,?), paid_at=COALESCE(paid_at,?) WHERE id=? AND token=?", (run, run or "TINYFISH", time.time(), cid, tok)).rowcount
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
            if now and not was: S["since"][s_] = seen.get(s_, vt); assign_vehicle(s_); open_visit(s_)
            if was and not now: _sec = S["last_seen"][s_] - S["since"].pop(s_, vt); new_charge(s_, _sec, S["vtype"].get(s_, "car")); visit_out(s_, _sec, "exit"); release_vehicle(s_)
            S["occ"][s_] = now
        S["overlay"] = {"moving": [tr["box"] for tr in cur if tr not in parked],
                        "slots": [(lay[s_], "booked" if booked_now(s_) else "occ" if S["occ"].get(s_) else "free",
                                   slot_label(i, j)) for i, row in enumerate(order) for j, s_ in enumerate(row)]}

def end_pass(vt, state):
    with LOCK:                                       # bill cars still parked, reset for the next loop
        for s_, was in list(S["occ"].items()):
            if was: _sec = max(0, vt - S["since"].pop(s_, vt)); new_charge(s_, _sec, S["vtype"].get(s_, "car")); visit_out(s_, _sec, "loop_end")
        S["occ"].clear(); S["since"].clear(); S["last_seen"].clear(); release_all(); S["loops"] = S.get("loops", 0) + 1
    state["tracks"].clear()
    if TRACKER:
        try: state["model"].predictor = None        # forget ByteTrack / BoT-SORT state for the next loop
        except Exception: pass

def placeholder(text):
    import numpy as np
    img = np.zeros((540, 960, 3), np.uint8); cv2.putText(img, text, (30, 270), 0, 0.9, (200, 200, 200), 2)
    return cv2.imencode(".jpg", img)[1].tobytes()

def player(lot):
    """Plays the video (file at real speed, or live webcam/RTSP) and ALWAYS streams frames, however slow detection is."""
    import traceback
    _tl.lot = lot                                    # this thread works on this lot's S / SH
    S["jpg"] = placeholder("Loading video...")
    live = VIDEO_SOURCE is not None and lot == FIRST
    try:
        vids = [] if live else lot_videos(lot)
        if not live and not vids:
            S["status"] = f"No video found in ./videos/{lot}"; S["jpg"] = placeholder(f"No video in ./videos/{lot}"); return
        COL = {"free": (0, 200, 0), "occ": (60, 60, 255), "booked": (255, 160, 0), "wrong": (255, 0, 200)}
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

# ---------------- two-camera lots: gate camera (left half) reads plates, overhead camera (right half) sees where cars park ----------------
DUAL_FILL = env_f("DUAL_FILL", 0.30)         # share of a slot that must differ from the empty lot to count as a car
DUAL_DIFF = env_f("DUAL_DIFF", 32)           # colour difference (0-255) that counts as "different from the empty lot"
DUAL_BASE = env_f("DUAL_BASE_SECS", 1.0)     # the lot is taken as empty for the first N video seconds of every loop (or put videos/lotN/empty.jpg)
DUAL_DEBUG = os.getenv("DUAL_DEBUG") == "1"  # print every slot's fill % on the video (for tuning DUAL_FILL / DUAL_DIFF)
ENTRY_TTL = env_f("ENTRY_TTL", 180)          # a plate read at the gate waits this long (video s) for its car to show up in a slot
WRONG_SECS = env_f("WRONG_SECS", 5)          # a car that does not match the slot's online booking must stay this long (s) before the guard is called
MIN_READS = max(1, int(env_f("MIN_READS", 2)))   # identical plate reads needed at the gate before a car is accepted
OCR_SCALE = env_f("OCR_SCALE", 2)            # enlarge the plate area before OCR
GUARD_PHONE = os.getenv("GUARD_PHONE")       # guard's mobile (10 digits or +91...); the call itself needs TWILIO_SID / TWILIO_TOKEN / TWILIO_FROM
GUARD_EMAIL = os.getenv("GUARD_EMAIL") or ADMIN_EMAIL

STATES = set("AN AP AR AS BR CG CH DD DL DN GA GJ HP HR JH JK KA KL LA LD MH ML MN MP MZ NL OD OR PB PY RJ SK TG TN TR TS UK UP WB".split())
_L2D = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "B": "8", "S": "5", "Z": "2", "G": "6"}   # letters OCR mistakes for digits
_D2L = {"0": "O", "1": "I", "8": "B", "5": "S", "2": "Z", "6": "G", "4": "A"}                       # digits OCR mistakes for letters

def fix_plate(raw):
    """OCR text -> Indian plate (SS DD AAA NNNN) with position-aware O/0, I/1, B/8, S/5 repairs, or None if it cannot be one."""
    t = re.sub(r"[^A-Z0-9]", "", (raw or "").upper())
    if not 8 <= len(t) <= 11: return None
    dig = lambda x: "".join(_L2D.get(ch, ch) if ch.isalpha() else ch for ch in x)
    let = lambda x, m: "".join(m.get(ch, ch) if ch.isdigit() else ch for ch in x)
    st, tail, mid = let(t[:2], {**_D2L, "1": "L"}), dig(t[-4:]), t[2:-4]
    if st not in STATES: st = let(t[:2], {**_D2L, "1": "L", "0": "D"})              # a 0 in a state code may be a D (D0 -> DL, etc.)
    if st not in STATES or not tail.isdigit() or not 3 <= len(mid) <= 5: return None
    best = None
    for d in (2, 1):                                  # 1 or 2 district digits, then 1-3 series letters
        a, b = dig(mid[:d]), let(mid[d:], {**_D2L, "1": "I"})
        if a.isdigit() and b.isalpha() and 1 <= len(b) <= 3:
            cost = sum(x != y for x, y in zip(a + b, mid))
            if best is None or cost < best[0]: best = (cost, a, b)
    return f"{st}{best[1]}{best[2]}{tail}" if best else None

def lev(a, b):
    if abs(len(a) - len(b)) > 2: return 9
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1): cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]

def colour_of(px):
    """N x 3 BGR pixels -> one of Red, Blue, Green, Yellow, White, Black, Orange (None if too few pixels)."""
    import numpy as np
    if px is None or len(px) < 20: return None
    hsv = cv2.cvtColor(px.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    h, s, v = hsv[:, 0].astype(int), hsv[:, 1].astype(int), hsv[:, 2].astype(int)
    chroma = (s > 70) & (v > 55)
    if chroma.mean() < 0.25: return "White" if np.median(v) > 120 else "Black"      # grey / silver counts as white
    hist = np.bincount(h[chroma], minlength=180).astype(float)
    hue = int(np.convolve(np.r_[hist[-3:], hist, hist[:3]], np.ones(7), "valid").argmax())   # smoothed, wraps around at red
    if hue < 8 or hue >= 170: return "Red"
    if hue < 22: return "Orange"
    if hue < 36: return "Yellow"
    if hue < 90: return "Green"
    return "Blue"

def make_ocr():
    """Plate reader: easyocr (pip install easyocr) or pytesseract. Returns (function, name); function(image) -> [(text, (x1, y1, x2, y2))]."""
    allow = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    try:
        import easyocr
        rd = easyocr.Reader(["en"], gpu=False, verbose=False)
        def run(im):
            out = []
            for pts, txt, cf in rd.readtext(im, allowlist=allow, detail=1):
                if cf >= 0.25:
                    xs, ys = [p[0] for p in pts], [p[1] for p in pts]; out.append((txt, (min(xs), min(ys), max(xs), max(ys))))
            return out
        return run, "easyocr"
    except Exception: pass
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        def run(im):
            d = pytesseract.image_to_data(im, config=f"--psm 11 -c tessedit_char_whitelist={allow}", output_type=pytesseract.Output.DICT)
            return [(t, (x, y, x + w, y + h)) for t, x, y, w, h, cf in zip(d["text"], d["left"], d["top"], d["width"], d["height"], d["conf"]) if t.strip() and float(cf) >= 30]
        return run, "tesseract"
    except Exception: return None, "NONE - pip install easyocr"

def plate_from_reads(reads, known):
    """OCR fragments -> (plate, box). Tries every fragment and every pair (plates are often split over two lines / two blocks)."""
    rs = sorted(reads, key=lambda r: (r[1][1] + r[1][3]) / 2)
    cands = [(t, b) for t, b in rs]
    for i, (t, b) in enumerate(rs):
        for t2, b2 in rs[i + 1:i + 3]:
            if abs((b[1] + b[3]) / 2 - (b2[1] + b2[3]) / 2) < 2.2 * max(b[3] - b[1], b2[3] - b2[1], 1):
                bb = (min(b[0], b2[0]), min(b[1], b2[1]), max(b[2], b2[2]), max(b[3], b2[3]))
                first, second = (t, t2) if b[0] <= b2[0] or b[1] + 5 < b2[1] else (t2, t)
                cands += [(first + second, bb), (second + first, bb)]
    for t, b in cands:
        n = re.sub(r"[^A-Z0-9]", "", t.upper()); p = fix_plate(t) or (n if n in known else None)
        if p: return p, b
    return None, None

def read_gate(left, model, ocr, known, D):
    """Gate camera: find the cars, read each plate, take each car's colour. Returns ([(plate, colour)], car boxes)."""
    res = model(left, classes=[2, 5, 7], imgsz=640, conf=CONF, verbose=False)
    boxes = [tuple(int(v) for v in b.xyxy[0].tolist()) for r in res for b in r.boxes]
    boxes = [b for b in boxes if b[2] - b[0] >= 60 and b[3] - b[1] >= 40]
    regions = [(b, True) for b in boxes]
    if not boxes and time.time() - D["fb"] >= 1.0:       # YOLO saw no car: look for plate text in the whole gate view (about once a second)
        D["fb"] = time.time(); regions = [((0, 0, left.shape[1], left.shape[0]), False)]
    out = []
    for (x1, y1, x2, y2), is_car in regions:
        top = y1 + int(0.35 * (y2 - y1)) if is_car else y1          # the plate is in the lower part of the car
        crop = left[top:y2, x1:x2]
        if crop.size == 0: continue
        sc = OCR_SCALE
        big = cv2.resize(crop, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC) if sc != 1 else crop
        reads = [(t, (b[0] / sc + x1, b[1] / sc + top, b[2] / sc + x1, b[3] / sc + top)) for t, b in ocr(big)]
        plate, pb = plate_from_reads(reads, known)
        if not plate: continue
        if is_car:                                                    # bonnet / grille area of the car
            h_, w_ = y2 - y1, x2 - x1
            body = left[y1 + int(0.30 * h_):y1 + int(0.65 * h_), x1 + int(0.2 * w_):x1 + int(0.8 * w_)]
        else:                                                         # no car box: the area above the plate
            ph, pw = max(8, pb[3] - pb[1]), max(8, pb[2] - pb[0])
            body = left[max(0, int(pb[1] - 5 * ph)):max(1, int(pb[1] - ph)), max(0, int(pb[0] - 2 * pw)):int(pb[2] + 2 * pw)]
        out.append((plate, colour_of(body.reshape(-1, 3)) if body.size else None))
    return out, boxes

def identity_for(plate, colour):
    """A plate read at the gate -> the plate stored in `vehicles`. One wrong character is snapped to the registered plate;
    a plate nobody registered gets a fake owner + phone and is saved, so every car has details."""
    with db() as c:
        rows = {r[0] for r in c.execute("SELECT plate FROM vehicles")}
        if plate not in rows:
            near = sorted((lev(plate, p), p) for p in rows)
            if near and near[0][0] <= 1 and (len(near) < 2 or near[1][0] > near[0][0]): plate = near[0][1]
        if plate not in rows:
            nm = f"{random.choice(FAKE_FIRST)} {random.choice(FAKE_LAST)}"
            c.execute("INSERT OR IGNORE INTO vehicles(plate,owner,phone,email,colour,fake) VALUES(?,?,?,?,?,1)",
                      (plate, nm, f"90000{random.randint(0, 99999):05d}", nm.lower().replace(" ", ".") + "@example.com", colour or random.choice(COLOURS)))
        elif colour: c.execute("UPDATE vehicles SET colour=? WHERE plate=? AND (fake=1 OR COALESCE(colour,'')='')", (colour, plate))
    return plate

def bind_vehicle(slot, plate):
    with db() as c:
        c.execute("INSERT OR REPLACE INTO parked(slot,lot,plate,since) VALUES(?,?,?,?)", (slot, _cur(), plate, time.time()))
        c.execute("UPDATE visits SET plate=? WHERE lot=? AND slot=? AND status='OPEN'", (plate, _cur(), slot))      # plate read after the car parked
    S["who"][slot] = plate

def booking_for(slot):
    """The online booking that covers this slot right now (with the booker's registered plate), or None."""
    t = time.time()
    with db() as c:
        r = c.execute("SELECT b.id,b.user_id,u.name,u.phone,u.plate FROM bookings b JOIN users u ON u.id=b.user_id "
                      "WHERE b.slot=? AND b.start<=? AND b.end>? ORDER BY b.id DESC LIMIT 1", (slot, t, t)).fetchone()
    return dict(r) if r else None

def lock_entry(plate, colour, vt, D):
    """A car was read at the gate. If a car is already standing in a slot with a placeholder identity and the same colour,
    it is that car; otherwise the plate waits in the queue for its car to show up in a slot."""
    plate = identity_for(plate, colour)
    ub = sorted((S["since"].get(s, vt), s) for s in D["unbound"] if S["occ"].get(s) and (not colour or D["colour"].get(s) in (None, colour)))
    if ub:
        s = ub[0][1]; D["unbound"].discard(s)
        with LOCK: bind_vehicle(s, plate)
    else: D["entries"].append({"plate": plate, "colour": colour, "t": vt})

def take_entry(D, colour, vt):
    """Oldest waiting gate read with the same colour; else oldest with unknown colour; else the oldest one (colour was misread)."""
    live = [e for e in D["entries"] if not e.get("used") and vt - e["t"] <= ENTRY_TTL]
    return (next((e for e in live if colour and e["colour"] == colour), None)
            or next((e for e in live if not e["colour"] or not colour), None) or (live[0] if live else None))

def dual_park(s, cand, vt, D):
    cols = cand["cols"]; col = max(set(cols), key=cols.count) if cols else None
    e = take_entry(D, col, vt)
    with LOCK:
        S["occ"][s] = True; S["since"][s] = cand["since"]; S["last_seen"][s] = vt; S["vtype"][s] = "car"; D["colour"][s] = col
        if e: e["used"] = True; bind_vehicle(s, e["plate"]); D["unbound"].discard(s)
        else: assign_vehicle(s); D["unbound"].add(s)          # plate not read (yet): placeholder identity so the grid still shows details
        open_visit(s)
    if col and S["who"].get(s):
        with db() as c: c.execute("UPDATE vehicles SET colour=? WHERE plate=? AND (fake=1 OR COALESCE(colour,'')='')", (col, S["who"][s]))

def dual_leave(s, vt, D):
    with LOCK:
        secs = max(0, S["last_seen"].get(s, vt) - S["since"].pop(s, vt))
        cov_ = D["covered"].pop(s, False)
        if not cov_: new_charge(s, secs, "car")
        visit_out(s, secs, "exit", covered=cov_)      # a car in its own booked slot already paid when it booked
        release_vehicle(s); S["occ"][s] = False; S["wrong"].pop(s, None)
        D["wsince"].pop(s, None); D["unbound"].discard(s); D["colour"].pop(s, None)

def guard_number():
    n = (os.getenv("GUARD_PHONE") or "").strip()
    if n.startswith("+"): return "+" + re.sub(r"\D", "", n)
    d = norm_phone(n)
    return "+91" + d if d else None

def guard_call(aid, text):
    """Phone the guard (Twilio voice call reading the alert out). Without Twilio settings the alert is only logged and e-mailed."""
    num, sid, tok, frm = guard_number(), os.getenv("TWILIO_SID"), os.getenv("TWILIO_TOKEN"), os.getenv("TWILIO_FROM")
    if not (num and sid and tok and frm): status = "LOGGED - set GUARD_PHONE + TWILIO_SID / TWILIO_TOKEN / TWILIO_FROM to place the call"
    else:
        say = xml_escape(text)
        try:
            r = requests.post(f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Calls.json", auth=(sid, tok), timeout=20,
                              data={"To": num, "From": frm, "Twiml": f'<Response><Say language="en-IN">{say}</Say><Pause length="1"/><Say language="en-IN">{say}</Say></Response>'})
            status = "CALLED" if r.status_code in (200, 201) else f"CALL FAILED ({r.status_code})"
        except Exception as e: status = "CALL FAILED: " + str(e)[:60]
    with db() as c: c.execute("UPDATE alerts SET status=? WHERE id=?", (status, aid))
    print(f"guard alert #{aid}: {status}")

def raise_alert(slot, b, plate):
    """Wrong parking: the car in this slot is not the one registered online for it. One alert per booking + car."""
    label = slot_labels().get(slot, f"#{slot}"); lot = _cur()
    v = vehicle_info([plate]).get(plate) or {}
    with db() as c:
        if c.execute("SELECT 1 FROM alerts WHERE booking_id=? AND plate=?", (b["id"], plate)).fetchone(): return
        aid = c.execute("INSERT INTO alerts(ts,lot,slot,label,plate,expected_plate,booking_id,booked_by,status) VALUES(?,?,?,?,?,?,?,?,?)",
                        (time.time(), lot, slot, label, plate, b["plate"], b["id"], b["name"], "CALLING")).lastrowid
    text = (f"Wrong parking at slot {label}, {lot}. This slot is booked online by {b['name']}, plate {' '.join(b['plate'])}. "
            f"A {v.get('colour') or ''} car, plate {' '.join(plate)}, is parked there.")
    threading.Thread(target=guard_call, args=(aid, text), daemon=True).start()
    notify(f"Wrong parking at {label} ({lot})", text + f"\nParked car owner: {v.get('owner', '?')}, phone {v.get('phone', '?')}.\nBooked by: {b['name']}, phone {b['phone']}.", GUARD_EMAIL)

def dual_guard(D, lay):
    """Every occupied slot with an online booking: the parked car's plate must be the booker's registered plate."""
    now = time.time()
    for s in list(lay):
        plate = S["who"].get(s)
        if not S["occ"].get(s) or not plate or s in D["unbound"]:          # no confirmed identity -> nothing to compare
            S["wrong"].pop(s, None); D["wsince"].pop(s, None); continue
        b = booking_for(s)
        if not b or not b["plate"]: S["wrong"].pop(s, None); D["wsince"].pop(s, None); continue
        if b["plate"] == plate:
            D["covered"][s] = True; S["wrong"].pop(s, None); D["wsince"].pop(s, None); continue
        t0 = D["wsince"].setdefault(s, now)
        if now - t0 >= WRONG_SECS and s not in S["wrong"]:
            S["wrong"][s] = b["id"]; raise_alert(s, b, plate)

def dual_step(frame, vt, D, model, ocr):
    import numpy as np
    H, W = frame.shape[:2]; left = frame[:, :W // 2]; boxes = []
    # ---- gate camera (left): plate + colour of every car in view; a plate is accepted after MIN_READS agreeing reads ----
    if model is not None and ocr is not None:
        with db() as c: known = {r[0] for r in c.execute("SELECT plate FROM vehicles")}
        reads, boxes = read_gate(left, model, ocr, known, D)
        for plate, colour in reads:
            key = next((k for k in D["votes"] if lev(k, plate) <= 1), plate)          # noisy reads of one plate vote together
            v = D["votes"].setdefault(key, {"vars": [], "cols": [], "last": vt}); v["vars"].append(plate); v["last"] = vt
            if colour: v["cols"].append(colour)
        for key, v in list(D["votes"].items()):
            if vt - v["last"] > 4: del D["votes"][key]; continue
            if len(v["vars"]) >= MIN_READS:
                del D["votes"][key]; plate = max(set(v["vars"]), key=v["vars"].count)
                if plate in S["who"].values() or vt - D["done"].get(plate, -1e9) < 30: continue
                D["done"][plate] = vt
                lock_entry(plate, max(set(v["cols"]), key=v["cols"].count) if v["cols"] else None, vt, D)
    # ---- overhead camera (right): which slots hold a car, and its colour ----
    lay, order = layout()
    ref = D["empty"]
    if ref is None:
        p = os.path.join(LOT_DIRS[_cur()], "empty.jpg"); img = cv2.imread(p) if os.path.exists(p) else None
        ref = D["empty"] = img if img is not None else False
    if ref is not False:
        if ref.shape[:2] != (H, W): ref = D["empty"] = cv2.resize(ref, (W, H))
        bg = ref
    else:
        if "t0" not in D: D["t0"] = vt; D["frames"] = []
        if D.get("bg") is None:
            D["frames"] = (D["frames"] + [frame.copy()])[-8:]
            if vt - D["t0"] >= DUAL_BASE: D["bg"] = np.median(np.stack(D["frames"]), axis=0).astype(np.uint8); D["frames"] = []
        bg = D.get("bg")
    if bg is not None:
        if D.get("bgb") is None: D["bgb"] = cv2.GaussianBlur(bg, (5, 5), 0)
        fb = cv2.GaussianBlur(frame, (5, 5), 0)
        for s, (x1, y1, x2, y2) in lay.items():
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
            if x2 - x1 < 8 or y2 - y1 < 8: continue
            m = cv2.absdiff(fb[y1:y2, x1:x2], D["bgb"][y1:y2, x1:x2]).max(axis=2) > DUAL_DIFF
            fill = float(m.mean()); D["fill"][s] = fill
            cand, occ = D["cand"].get(s), S["occ"].get(s)
            if fill >= DUAL_FILL:
                if cand is None: cand = D["cand"][s] = {"since": vt, "cols": []}
                cand["last"] = vt
                col = colour_of(frame[y1:y2, x1:x2][m])
                if col: cand["cols"] = (cand["cols"] + [col])[-12:]
                if occ: S["last_seen"][s] = vt
                elif vt - cand["since"] >= PARK_SECS: dual_park(s, cand, vt, D)      # still for PARK_SECS -> parked
            elif cand and vt - cand["last"] >= (OCC_GRACE if occ else 1.0):
                D["cand"].pop(s)
                if occ: dual_leave(s, vt, D)
    for s in [s for s in D["cand"] if s not in lay]: D["cand"].pop(s)               # slot deleted in the editor
    dual_guard(D, lay)
    with LOCK:
        sl = []
        for i, row in enumerate(order):
            for j, s in enumerate(row):
                st = "wrong" if s in S["wrong"] else {"occupied": "occ", "booked": "booked", "free": "free"}[slot_state(s)]
                sl.append((lay[s], st, slot_label(i, j) + (f" {D['fill'].get(s, 0):.0%}" if DUAL_DEBUG else "")))
        S["overlay"] = {"moving": boxes, "slots": sl}

def dual_end_pass(vt, D):
    """The video looped: bill cars still parked, forget everything and measure the empty lot again."""
    with LOCK:
        for s_, was in list(S["occ"].items()):
            if was:
                _sec = max(0, vt - S["since"].pop(s_, vt)); cov_ = D["covered"].pop(s_, False)
                if not cov_: new_charge(s_, _sec, "car")
                visit_out(s_, _sec, "loop_end", covered=cov_)
        S["occ"].clear(); S["since"].clear(); S["last_seen"].clear(); S["wrong"].clear(); release_all(); S["loops"] = S.get("loops", 0) + 1
    for k in ("cand", "entries", "votes", "done", "wsince", "covered", "colour"): D[k].clear()
    D["unbound"].clear()
    for k in ("bg", "bgb", "t0", "frames"): D.pop(k, None)

def dual_detector(lot):
    import traceback
    _tl.lot = lot
    S["wrong"] = {}
    D = {"empty": None, "cand": {}, "entries": [], "votes": {}, "done": {}, "wsince": {}, "covered": {}, "colour": {}, "fill": {},
         "unbound": set(), "fb": 0}
    model = None
    try:
        from ultralytics import YOLO
        model = YOLO(YOLO_MODEL)
    except Exception as e: print("gate detector unavailable:", e)
    ocr, oname = make_ocr()
    note = f"gate: yolo {'on' if model else 'OFF'}, ocr {oname}"
    S["det"] = "starting - " + note
    last = 0
    while True:
        if SH["pass_end"] is not None:
            v = SH["pass_end"]; SH["pass_end"] = None
            try: dual_end_pass(v, D)
            except Exception: traceback.print_exc()
            continue
        if SH["frame"] is None or SH["seq"] == last: time.sleep(0.02); continue
        last = SH["seq"]; t_ = time.time()
        try: dual_step(SH["frame"], SH["vt"], D, model, ocr); S["det"] = "detecting - " + note
        except Exception as e: traceback.print_exc(); S["det"] = "error: " + repr(e)[:80]
        S["det_ms"] = int((time.time() - t_) * 1000)
        time.sleep(max(0.05, 0.25 - (time.time() - t_)))

def detector(lot):
    import traceback
    _tl.lot = lot
    if lot in DUAL: return dual_detector(lot)       # two-camera footage: plate OCR (left) + colour grid (right)
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
            with db() as c:
                for k in LOTS:
                    with using(k), LOCK: st = [slot_state(s) for s in S["slots"]]
                    if st: c.execute("INSERT INTO occ_log(ts,free,occupied,booked,lot) VALUES(?,?,?,?,?)", (time.time(), st.count("free"), st.count("occupied"), st.count("booked"), k))
                c.execute("DELETE FROM occ_log WHERE ts<?", (time.time() - 30 * 86400,))
        except Exception as e: print("history log error:", e)

SNAP_EVERY = env_f("SNAP_EVERY", 5)
def snapshot_live():
    """Write who is parked right now (plate, slot, seconds so far) so a crash or restart cannot make the stay vanish unbilled."""
    L = live_sessions()
    with db() as c:
        c.execute("DELETE FROM live_snap")
        c.executemany("INSERT INTO live_snap(lot,slot,plate,vtype,seconds,updated) VALUES(?,?,?,?,?,?)", [(x["lot"], x["slot"], x["plate"], x["vtype"], x["seconds"], time.time()) for x in L])

def snapshot_loop():
    while True:
        time.sleep(SNAP_EVERY)
        try: snapshot_live()
        except Exception as e: print("snapshot error:", e)

def recover_sessions():
    """On start: stays that were still running when the app stopped are billed for the time they had so far (once), then the snapshot is cleared."""
    with db() as c: rows = [dict(r) for r in c.execute("SELECT * FROM live_snap")]; c.execute("DELETE FROM live_snap")
    n = 0
    for r in rows:
        if r["lot"] not in LOTS or r["seconds"] <= 0: continue
        with using(r["lot"]):
            S["who"][r["slot"]] = r["plate"]
            new_charge(r["slot"], r["seconds"], r["vtype"] or "car"); visit_out(r["slot"], r["seconds"], "restart_recovered")
            S["who"].pop(r["slot"], None)
        n += 1
    with db() as c: c.execute("UPDATE visits SET status='CLOSED', left=?, source='restart_unclosed' WHERE status='OPEN'", (time.time(),))      # entry kept, exit unknown
    if n: print(f"recovered {n} stay(s) that were running when the app stopped - billed for their time so far")

_started = False
def start_workers():
    global _started
    if _started: return
    _started = True
    for k in LOTS: print(f"[{k}] folder {LOT_DIRS[k]!r}: {[os.path.basename(v) for v in lot_videos(k)] or 'NO VIDEO FOUND'}" + ("  (two-camera lot)" if k in DUAL else ""))
    for k in LOTS:
        for fn in (player, detector): threading.Thread(target=fn, args=(k,), daemon=True).start()
    threading.Thread(target=logger_loop, daemon=True).start()
    if REMIND_MINS > 0: threading.Thread(target=remind_loop, daemon=True).start()
    threading.Thread(target=snapshot_loop, daemon=True).start()

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
def pick_lot():                                      # every request works on the lot named in ?lot= (default lot1)
    _tl.lot = request.args.get("lot") if request.args.get("lot") in LOTS else FIRST

def norm_phone(s):
    d = re.sub(r"\D", "", s or "")
    if len(d) == 12 and d.startswith("91"): d = d[2:]
    if len(d) == 11 and d.startswith("0"): d = d[1:]
    return d if len(d) == 10 and d[0] in "6789" else None

def norm_plate(s):
    p = re.sub(r"[^A-Z0-9]", "", (s or "").upper())
    return p if 6 <= len(p) <= 12 and re.search(r"[A-Z]", p) and re.search(r"\d", p) else None

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
<form id=f-register method=post action=/auth/register><input name=name placeholder="Full name" required><input name=email type=email placeholder=Email required><input name=phone type=tel inputmode=numeric placeholder="Phone number (10 digits)" required><input name=plate placeholder="Vehicle number plate (e.g. DL8CAF7305)" required style="text-transform:uppercase"><input name=pw type=password minlength=6 placeholder="Password (min 6 characters)" required><button class=go>Create account</button></form>
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
    phone, plate = norm_phone(f.get("phone", "")), norm_plate(f.get("plate", ""))
    if not phone: return render_auth("register", "Enter a valid 10-digit mobile number")
    if not plate: return render_auth("register", "Enter your vehicle number plate, e.g. DL8CAF7305")
    note_fail(key)
    try:
        with db() as c:
            if c.execute("SELECT 1 FROM vehicles WHERE plate=?", (plate,)).fetchone(): return render_auth("register", "That number plate is already registered")
            first = c.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0
            c.execute("INSERT INTO users(name,email,pw,role,balance,phone,plate) VALUES(?,?,?,?,?,?,?)",
                      (name, email, generate_password_hash(pw), "admin" if first else "user", START_BALANCE if WALLET else 0, phone, plate))
            u = c.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
            c.execute("INSERT INTO vehicles(plate,owner,phone,email,fake,user_id) VALUES(?,?,?,?,0,?)", (plate, name, phone, email, u["id"]))
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
        V = vehicle_info(S["who"].values())
        rows = [[{"id": s, "label": slot_label(i, j), "who": who_view(V.get(S["who"].get(s)), adm),
                  "state": slot_state(s), "wrong": bool(adm and s in (S.get("wrong") or {}))}
                 for j, s in enumerate(r)] for i, r in enumerate(layout()[1])]
    status = (S["status"] + " | " + S.get("det", "")) if adm else ("online" if S["status"].startswith("live") else S["status"])
    out = {"rows": rows, "status": status, "vt": round(S["vt"], 1), "rate": RATE, "admin": adm, "pricing": pricing_text(), "auto": S["auto"], "lot": _cur()}
    with db() as c:
        if adm:
            out["charges"] = [dict(r) for r in c.execute("SELECT id,slot,seconds,amount,status,run,vtype,plate FROM charges ORDER BY id DESC LIMIT 15")]
            out["revenue"] = c.execute("SELECT COALESCE(SUM(amount),0) FROM charges WHERE status='PAID'").fetchone()[0]
        else: out["charges"], out["revenue"] = [], None
        out["mine"] = [dict(r) for r in c.execute("SELECT id,slot,start,end,amount FROM bookings WHERE user_id=? AND end>? ORDER BY start", (session["uid"], time.time()))]
        if WALLET:
            out["balance"] = c.execute("SELECT balance FROM users WHERE id=?", (session["uid"],)).fetchone()[0] or 0
            out["claims"] = [r["slot"] for r in c.execute("SELECT slot FROM claims WHERE user_id=? AND created>?", (session["uid"], time.time() - CLAIM_TTL))]
    return jsonify(out)

def slot_labels():
    out = {}
    for k in LOTS:
        with using(k), LOCK: out.update({s: slot_label(i, j) for i, r in enumerate(layout()[1]) for j, s in enumerate(r)})
    return out

def slot_counts(lot=None):
    with using(lot or _cur()), LOCK: st = [slot_state(s) for s in S["slots"]]
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
    adm = session.get("role") == "admin"
    with db() as c: rows = [dict(r) for r in c.execute("SELECT * FROM parkings" + ("" if adm else " WHERE COALESCE(status,'active')='active'"))]
    for r in rows:
        r["km"] = round(haversine(lat, lng, r["lat"], r["lng"]), 2) if have else None
        r["rate"] = RATE if r["live"] or r["rate"] is None else r["rate"]       # rupees per second shown to drivers
        if r["live"]: cnt = slot_counts(r["lot"] if r["lot"] in LOTS else FIRST); r["free"], r["total"] = cnt["free"], cnt["total"]
    rows.sort(key=lambda r: (r["km"] is None, r["km"] or 0, r["name"]))
    out = dict(parkings=rows, located=have)
    if adm: out["discover"] = X.get("discover", "")
    return jsonify(**out)

def parking_fields(d):
    name, address = str(d.get("name", "")).strip()[:80], str(d.get("address", "")).strip()[:160]
    lat, lng = float(d["lat"]), float(d["lng"])
    rate = d.get("rate"); rate = None if rate in (None, "") else float(rate)
    if not name or not (-90 <= lat <= 90 and -180 <= lng <= 180) or (rate is not None and not 0 < rate <= 1000): raise ValueError
    return name, address, lat, lng, rate

@app.route("/api/parkings", methods=["POST"])
@need_admin
def parkings_add():
    try: f = parking_fields(jd())
    except (KeyError, ValueError, TypeError): return jsonify(error="Enter a name and a valid latitude / longitude"), 400
    with db() as c: pid = c.execute("INSERT INTO parkings(name,address,lat,lng,live,rate) VALUES(?,?,?,?,0,?)", f).lastrowid
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
        c.execute("UPDATE parkings SET name=?, address=?, lat=?, lng=?, rate=? WHERE id=?", (*f, pid))
    return jsonify(ok=True)

@app.route("/api/admin/bookings")
@need_admin
def admin_bookings():
    lab = slot_labels()
    with db() as c:
        rows = [dict(r) for r in c.execute("SELECT b.id,b.slot,b.start,b.end,b.amount,u.name,u.email FROM bookings b LEFT JOIN users u ON u.id=b.user_id ORDER BY b.start DESC LIMIT 200")]
    for r in rows: r["label"] = lab.get(r["slot"], f"#{r['slot']}")
    return jsonify(bookings=rows)

@app.route("/api/alerts")
@need_admin
def alerts_list():
    with db() as c: rows = [dict(r) for r in c.execute("SELECT * FROM alerts ORDER BY id DESC LIMIT 20")]
    return jsonify(alerts=rows)

@app.route("/api/alerts/<int:aid>/resolve", methods=["POST"])
@need_admin
def alerts_resolve(aid):
    with db() as c: c.execute("UPDATE alerts SET status='RESOLVED - ' || status WHERE id=? AND status NOT LIKE 'RESOLVED%'", (aid,))
    return jsonify(ok=True)

@app.route("/api/book", methods=["POST"])
@need_login
def book():
    d = jd()
    try: slot = int(d["slot"]); start_in = max(0, float(d.get("start_min", 0))) * 60; secs = float(d["seconds"])
    except (KeyError, ValueError, TypeError): return jsonify(error="invalid"), 400
    method = d.get("method") or ("wallet" if WALLET else "gateway")
    if method not in ("wallet", "gateway") or (method == "wallet" and not WALLET): return jsonify(error="invalid payment method"), 400
    if secs <= 0 or secs > 86400 or slot not in S["slots"]: return jsonify(error="invalid"), 400
    s0 = time.time() + start_in; e0 = s0 + secs; amt = price(secs)      # the amount is always worked out here, on the server
    cid = tok = None
    with db() as c:
        if c.execute("SELECT 1 FROM bookings WHERE slot=? AND start<? AND end>?", (slot, e0, s0)).fetchone():
            return jsonify(error="Slot already booked for that time"), 409
        if start_in == 0 and S["occ"].get(slot): return jsonify(error="Slot is occupied"), 409
        if method == "wallet" and c.execute("UPDATE users SET balance=balance-? WHERE id=? AND balance>=?", (amt, session["uid"], amt)).rowcount == 0:
            return jsonify(error="Not enough wallet balance - top up or pay through the gateway"), 402
        bid = c.execute("INSERT INTO bookings(user_id,slot,start,end,amount) VALUES(?,?,?,?,?)", (session["uid"], slot, s0, e0, amt)).lastrowid
        if method == "wallet": wtx(c, session["uid"], "BOOKING", -amt, f"booking #{bid}")
        if method == "gateway" and amt > 0:      # slot is held while the payment page is filled in and paid; released if it fails
            tok = secrets.token_urlsafe(12)
            cid = c.execute("INSERT INTO charges(slot,seconds,amount,status,token,created,vtype,user_id,booking_id) VALUES(?,?,?,?,?,?,?,?,?)",
                            (slot, int(round(secs)), amt, "PENDING", tok, time.time(), "car", session["uid"], bid)).lastrowid
        u = c.execute("SELECT email FROM users WHERE id=?", (session["uid"],)).fetchone()
    if cid: threading.Thread(target=tinyfish_pay, args=(cid, tok, amt), daemon=True).start()
    elif u: notify("Parking booking confirmed", f"Slot {slot}: {time.strftime('%d %b %H:%M', time.localtime(s0))} for {int(secs)} s, \u20b9{amt}.", u["email"])
    return jsonify(ok=True, charged=amt, id=bid, charge_id=cid)

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
        ch = c.execute("SELECT status FROM charges WHERE booking_id=?", (bid,)).fetchone()
        ref = refund_amount(b) if WALLET and (not ch or ch["status"] == "PAID") else 0     # unpaid gateway bookings refund nothing
        if ch and ch["status"] == "PENDING": c.execute("UPDATE charges SET status='FAILED', run='booking cancelled' WHERE booking_id=?", (bid,))
        c.execute("DELETE FROM bookings WHERE id=?", (bid,))
        if ref: c.execute("UPDATE users SET balance=balance+? WHERE id=?", (ref, b["user_id"])); wtx(c, b["user_id"], "REFUND", ref, f"booking #{bid} cancelled")
    return jsonify(ok=True, refunded=ref)

@app.route("/api/booking/<int:bid>", methods=["PUT"])
@need_login
def edit_booking(bid):
    d = jd(); now = time.time()
    with db() as c:
        b = c.execute("SELECT * FROM bookings WHERE id=?", (bid,)).fetchone()
        if not b or (b["user_id"] != session["uid"] and session.get("role") != "admin"): return jsonify(error="not found"), 404
        if b["start"] <= now: return jsonify(error="Only bookings that have not started yet can be edited"), 409
        if c.execute("SELECT 1 FROM charges WHERE booking_id=?", (bid,)).fetchone():
            return jsonify(error="Bookings paid through the gateway cannot be edited - cancel and book again"), 409
        try:
            start_in = max(0, float(d["start_min"])) * 60 if "start_min" in d else b["start"] - now
            secs = float(d.get("seconds", b["end"] - b["start"]))
        except (ValueError, TypeError): return jsonify(error="invalid"), 400
        if secs <= 0 or secs > 86400: return jsonify(error="invalid"), 400
        s0 = now + start_in; e0 = s0 + secs
        if c.execute("SELECT 1 FROM bookings WHERE slot=? AND id<>? AND start<? AND end>?", (b["slot"], bid, e0, s0)).fetchone():
            return jsonify(error="Slot already booked for that time"), 409
        if start_in == 0 and slot_busy(b["slot"]): return jsonify(error="Slot is occupied"), 409
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
    if RZP and not FREE_TOPUP: return jsonify(error="Top up through Razorpay (My payments page)"), 400
    try: amt = float(jd().get("amount", 0))
    except (ValueError, TypeError): return jsonify(error="invalid"), 400
    if not 1 <= amt <= 10000: return jsonify(error="amount must be between 1 and 10000"), 400
    with db() as c:                                  # test mode: credit instantly (plug a real gateway in here)
        c.execute("UPDATE users SET balance=balance+? WHERE id=?", (amt, session["uid"])); wtx(c, session["uid"], "TOPUP", amt, "test credit")
        bal = c.execute("SELECT balance FROM users WHERE id=?", (session["uid"],)).fetchone()[0]
    return jsonify(ok=True, balance=bal)

@app.route("/api/claim", methods=["POST"])
@need_login
def claim():
    """'Bill my wallet': when the car in this slot leaves, the time-based charge is deducted from this driver's wallet."""
    if not WALLET: return jsonify(error="wallet is disabled (set WALLET=1)"), 404
    d = jd()
    try: slot = int(d["slot"])
    except (KeyError, ValueError, TypeError): return jsonify(error="invalid"), 400
    if slot not in S["slots"]: return jsonify(error="no such slot"), 404
    with db() as c:
        if d.get("off"):
            c.execute("DELETE FROM claims WHERE slot=? AND user_id=?", (slot, session["uid"])); return jsonify(ok=True)
        cur = c.execute("SELECT user_id FROM claims WHERE slot=? AND created>? ORDER BY id DESC LIMIT 1", (slot, time.time() - CLAIM_TTL)).fetchone()
        if cur and cur["user_id"] != session["uid"]: return jsonify(error="Another driver is already billing this slot"), 409
        if not cur: c.execute("INSERT INTO claims(user_id,slot,created) VALUES(?,?,?)", (session["uid"], slot, time.time()))
    return jsonify(ok=True)

@app.route("/api/charge/<int:cid>")
@need_login
def charge_status(cid):
    with db() as c: ch = c.execute("SELECT status, user_id FROM charges WHERE id=?", (cid,)).fetchone()
    if not ch or (ch["user_id"] != session["uid"] and session.get("role") != "admin"): return jsonify(error="not found"), 404
    return jsonify(status=ch["status"])

def discover_job(lat, lng):
    """TinyFish agent searches the web for parkings near the live one; results are saved as 'review' (hidden from drivers until approved)."""
    X["discover"] = "TinyFish is searching the web..."
    try:
        goal = (f"Find up to 8 car parking lots or parking facilities within about 2 km of latitude {lat}, longitude {lng} (Dwarka, New Delhi). "
                f"For each one give its name, address, latitude and longitude exactly as the site shows them - never guess coordinates. "
                f"Return JSON {{\"parkings\": [{{\"name\": \"\", \"address\": \"\", \"lat\": 0, \"lng\": 0}}]}}.")
        res = tf_run(DISCOVER_URL, goal, 600).get("result")
        items = res.get("parkings") if isinstance(res, dict) else res
        n = 0
        with db() as c:
            have = [dict(r) for r in c.execute("SELECT name,lat,lng FROM parkings")]
            for it in (items if isinstance(items, list) else [])[:12]:
                try: name, la, lo = str(it["name"]).strip()[:80], float(it["lat"]), float(it["lng"]); addr = str(it.get("address", "")).strip()[:160]
                except (KeyError, ValueError, TypeError, AttributeError): continue
                if not name or not (-90 <= la <= 90 and -180 <= lo <= 180) or haversine(lat, lng, la, lo) > 5: continue
                if any(h["name"].lower() == name.lower() or haversine(h["lat"], h["lng"], la, lo) < 0.03 for h in have): continue
                c.execute("INSERT INTO parkings(name,address,lat,lng,live,status) VALUES(?,?,?,?,0,'review')", (name, addr, la, lo))
                have.append({"name": name, "lat": la, "lng": lo}); n += 1
        X["discover"] = f"Found {n} new parking(s) - approve or delete them below." if n else "Nothing new found."
    except Exception as e: X["discover"] = "Discovery failed: " + str(e)[:80]

@app.route("/api/parkings/discover", methods=["POST"])
@need_admin
def parkings_discover():
    if not TF_KEY: return jsonify(error="Set TINYFISH_API_KEY first"), 400
    if X.get("discover", "").startswith("TinyFish is"): return jsonify(error="A search is already running"), 409
    with db() as c: lv = c.execute("SELECT lat,lng FROM parkings WHERE live=1").fetchone()
    if not lv: return jsonify(error="no live parking"), 404
    X["discover"] = "TinyFish is searching the web..."
    threading.Thread(target=discover_job, args=(lv["lat"], lv["lng"]), daemon=True).start()
    return jsonify(ok=True)

@app.route("/api/parkings/<int:pid>/approve", methods=["POST"])
@need_admin
def parkings_approve(pid):
    with db() as c: n = c.execute("UPDATE parkings SET status='active' WHERE id=?", (pid,)).rowcount
    return jsonify(ok=True) if n else (jsonify(error="not found"), 404)

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
        if ch["booking_id"] and not c.execute("SELECT 1 FROM bookings WHERE id=?", (ch["booking_id"],)).fetchone():
            return jsonify(error="That booking was released after the payment failed - book again"), 409
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
        elif request.args.get("kind") == "plates":
            w.writerow(["plate", "owner", "phone", "stays", "seconds", "billed", "paid", "pending", "failed"])
            for r in c.execute("SELECT COALESCE(ch.plate,'UNKNOWN') p, v.owner, v.phone, COUNT(*), SUM(ch.seconds), SUM(ch.amount), SUM(CASE WHEN ch.status='PAID' THEN ch.amount ELSE 0 END), "
                               "SUM(CASE WHEN ch.status='PENDING' THEN ch.amount ELSE 0 END), SUM(CASE WHEN ch.status='FAILED' THEN ch.amount ELSE 0 END) "
                               "FROM charges ch LEFT JOIN vehicles v ON v.plate=ch.plate GROUP BY p ORDER BY p"): w.writerow([csv_safe(x) for x in r])
            name = "payments_by_plate.csv"
        elif request.args.get("kind") == "visits":
            w.writerow(["id", "lot", "slot", "plate", "in", "out", "seconds", "video_loop", "app_start", "charge_id", "status", "source"])
            fmt = lambda t: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t)) if t else ""
            for r in c.execute("SELECT * FROM visits ORDER BY id"): w.writerow([r["id"], r["lot"], r["slot"], csv_safe(r["plate"]), fmt(r["entered"]), fmt(r["left"]), r["seconds"], r["loop_no"], r["boot"], r["charge_id"], r["status"], r["source"]])
            name = "in_out_log.csv"
        elif request.args.get("kind") == "wallet":
            w.writerow(["time", "driver", "plate", "kind", "amount", "ref", "balance_after"])
            for r in c.execute("SELECT w.ts,w.kind,w.amount,w.ref,w.balance,u.name,u.plate FROM wallet_tx w LEFT JOIN users u ON u.id=w.user_id ORDER BY w.id DESC"):
                w.writerow([time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r["ts"])), csv_safe(r["name"]), csv_safe(r["plate"]), r["kind"], r["amount"], csv_safe(r["ref"]), r["balance"]])
            name = "wallet_history.csv"
        elif request.args.get("kind") == "daily":
            w.writerow(["date", "charges", "seconds", "revenue_paid"])
            for r in c.execute("SELECT date(created,'unixepoch','localtime') d, COUNT(*), SUM(seconds), SUM(CASE WHEN status='PAID' THEN amount ELSE 0 END) FROM charges GROUP BY d ORDER BY d DESC"): w.writerow(list(r))
            name = "revenue_by_day.csv"
        else:
            w.writerow(["id", "slot", "vehicle", "seconds", "amount", "status", "created", "run", "plate", "method"])
            for r in c.execute("SELECT id,slot,vtype,seconds,amount,status,created,run,plate,method FROM charges ORDER BY id"):
                w.writerow([r["id"], r["slot"], csv_safe(r["vtype"]), r["seconds"], r["amount"], r["status"],
                            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r["created"])), csv_safe(r["run"]), csv_safe(r["plate"]), csv_safe(r["method"])])
            name = "charges.csv"
    return Response(out.getvalue(), mimetype="text/csv", headers={"Content-Disposition": f"attachment; filename={name}"})

@app.route("/api/history")
@need_login
def history():
    with db() as c:
        rows = c.execute("SELECT strftime('%H', ts,'unixepoch','localtime') h, AVG(100.0*(occupied+booked)/(free+occupied+booked)) p "
                         "FROM occ_log WHERE free+occupied+booked>0 AND COALESCE(lot,'lot1')=? GROUP BY h", (_cur(),)).fetchall()
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
    u = db().execute("SELECT name,email FROM users WHERE id=?", (ch["user_id"],)).fetchone() if ch["user_id"] else None
    return render_template_string("""<!doctype html><title>Pay</title><body style="font-family:system-ui;max-width:360px;margin:40px auto">
    <h2>Parking payment #{{c.id}}</h2><p>Slot {{c.slot}} &middot; {{c.seconds}} s &middot; <b>Amount: {{c.amount}} rupees</b></p>
    {% if c.status=='PAID' %}<h3 id=ok>Payment successful</h3>{% else %}
    <form method=post action="/pay/{{c.id}}?t={{t}}"><input name=name value="{{u.name if u else 'Test User'}}" required><br><br><input name=email value="{{u.email if u else 'test@example.com'}}" required><br><br><input name=card value="4242 4242 4242 4242" placeholder="Card number" required><br><br>
    <input name=exp value="12/30" placeholder="MM/YY" required> <input name=cvv value="123" placeholder="CVV" required><br><br>
    <button id=pay-btn>Pay Now</button></form>{% endif %}""", c=ch, t=tok, u=u)

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
.who{font-size:11px;color:#cbd5e1;line-height:1.35;word-break:break-word}.who b{color:#fff;letter-spacing:.04em;font-size:12px}.slot b{font-size:13px}.slot .ic{font-size:26px}.slot small{font-size:11px;text-transform:uppercase;letter-spacing:.08em}
.free{background:#10b98118;border-color:#10b98177;color:#6ee7b7}.occupied{background:#ef444418;border-color:#ef444477;color:#fca5a5}.booked{background:#f59e0b18;border-color:#f59e0b77;color:#fcd34d}
.free:hover{transform:translateY(-2px);box-shadow:0 8px 24px #10b98133}
.wr{box-shadow:0 0 0 2px #fb7185,0 0 18px #fb718566}
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
<div class=user>{{name}}{% if is_admin %} (admin){% endif %}<span id=bal></span><a href=/nearby>Nearby parkings</a><a href=/bookings>My bookings</a><a href=/payments>My payments</a>{% if is_admin %}<a href=/admin>Admin</a><a href=/admin/payments>Payments</a>{% if lots|length>1 %}{% for k in lots %}<a href="/admin?lot={{k}}">{{k}}</a>{% endfor %}{% endif %}{% endif %}{% if admin %}<a href="/admin/export.csv?kind=bookings">Bookings CSV</a><a href="/admin/export.csv?kind=charges">Charges CSV</a><a href="/admin/export.csv?kind=daily">Daily CSV</a>{% endif %}<a href=/logout>Logout</a></div></header>
<div class=stats><div class=stat><small>Free</small><b id=sf style=color:#6ee7b7>0</b></div><div class=stat><small>Occupied</small><b id=so style=color:#fca5a5>0</b></div>
<div class=stat><small>Booked</small><b id=sb style=color:#fcd34d>0</b></div><div class=stat><small>Occupancy</small><b id=sp>-</b></div>{% if admin %}<div class=stat><small>Revenue collected</small><b id=sr>&#8377;0</b></div>{% endif %}</div>
<main><div class=card><h3>{{ptitle}} &middot; live camera</h3><div style="position:relative"><img id=cam alt="Live camera">{% if admin %}<div id=ov style="position:absolute;inset:0;display:none;cursor:crosshair"><div id=rb style="position:absolute;border:2px dashed #34d399;display:none;pointer-events:none"></div></div>{% endif %}</div><div class=info id=info></div>{% if admin %}<div style="display:flex;gap:8px;margin-top:10px;flex-wrap:wrap"><button id=tg onclick=tog()>Edit slots</button><button class=g onclick=autoT()>Auto-detect: <span id=au>on</span></button><button class=g onclick=clr()>Clear all slots</button></div><div class=info>Edit mode: drag on the video to add a slot, click a slot to delete it (to resize, delete and redraw).</div>{% endif %}</div>
<div><div class=card><h3>Parking grid</h3><div class=leg><span><i style=background:#10b981></i>Free</span><span><i style=background:#ef4444></i>Occupied</span><span><i style=background:#f59e0b></i>Booked</span></div>
<div id=grid class=info>Waiting for parked cars to be detected...</div></div>
{% if admin %}<div class=card style=margin-top:16px><h3>All bookings</h3><div style="overflow-x:auto"><table id=ab></table></div></div>
<div class=card style=margin-top:16px><h3>Latest charges &middot; <a href=/admin/payments style=color:#67e8f9>open full ledger</a></h3><table id=ch></table></div>
<div class=card style=margin-top:16px><h3>Wrong-parking alerts &middot; guard called</h3><div style="overflow-x:auto"><table id=al></table></div></div>
<div class=card style=margin-top:16px><h3>Parkings</h3><div class=info id=pdi></div><div style="overflow-x:auto"><table id=pk></table></div>
<div style="display:grid;gap:8px;margin-top:12px"><input id=pn placeholder="Name"><input id=pa placeholder="Address (optional)">
<div style="display:flex;gap:8px"><input id=pla placeholder=Latitude style="width:50%"><input id=plo placeholder=Longitude style="width:50%"></div><input id=prt placeholder="Price per second in rupees (blank = default)">
<div style="display:flex;gap:8px;flex-wrap:wrap"><button onclick=pHere()>Use my location</button><button onclick=pSave()><span id=psl>Add parking</span></button><button class=g onclick=pClear()>Clear</button><button class=g onclick=pFind()>Find nearby via TinyFish</button></div></div></div>{% endif %}<div class=card style=margin-top:16px>
<h3 style=margin-top:16px>My bookings</h3><div id=mine class=info></div></div>
<div class=card style=margin-top:16px><h3>Busy hours (heatmap)</h3><div id=hm style="display:grid;grid-template-columns:repeat(24,1fr);gap:3px"></div><div class=info id=hmi></div></div></div></main>
<div id=m><div><h3 id=mt></h3><label>Start in (minutes, 0 = now)<input id=sm type=number value=0 min=0></label>
<label>Duration (seconds)<input id=ds type=number value=60 min=1></label><div class=cost id=cost></div>
<label>Pay with<select id=pm style="padding:10px;border-radius:9px;border:1px solid var(--bd);background:#0b1020;color:#fff"><option value=wallet>Wallet</option><option value=gateway>Payment gateway (details filled in automatically)</option></select></label>
<button style="padding:12px;font-size:14px" onclick=pay()>Pay &amp; book</button><button class=g style="padding:10px;font-size:13px" onclick="m.style.display='none'">Cancel</button><div id=er></div></div></div>
<div id=t></div>
<script>const LOT='{{lot}}';{const f0=window.fetch;window.fetch=(u,o)=>f0(typeof u==='string'&&u[0]==='/'?u+(u.includes('?')?'&':'?')+'lot='+LOT:u,o)}let slot=null,rate=10,hasW=false;const H={'Content-Type':'application/json','X-CSRF-Token':'{{csrf}}'};const $=i=>document.getElementById(i),m=$('m');
function open_(id,l){slot=id;$('mt').textContent='Book slot '+l;m.style.display='grid';$('er').textContent='';$('pm').querySelector('option[value=wallet]').hidden=!hasW;$('pm').value=hasW?'wallet':'gateway';up()}
function up(){$('cost').textContent='Estimated total: \u20B9'+(+$('ds').value*rate)}$('ds').oninput=up;
function toast(x){const t=$('t');t.textContent=x;t.style.display='block';setTimeout(()=>t.style.display='none',3000)}
async function pay(){const r=await fetch('/api/book',{method:'POST',headers:H,body:JSON.stringify({slot,start_min:$('sm').value,seconds:$('ds').value,method:$('pm').value})});
const j=await r.json();if(!r.ok){$('er').textContent=j.error||'failed';return}m.style.display='none';
if(j.charge_id){toast('Payment in progress...');poll(j.charge_id)}else toast('Booked! Charged \u20B9'+j.charged);tick()}
async function poll(id){for(let i=0;i<120;i++){await new Promise(r=>setTimeout(r,3000));const j=await (await fetch('/api/charge/'+id)).json();
if(j.status=='PAID'){toast('Payment received - booking confirmed');tick();return}
if(j.status=='FAILED'){toast('Payment failed - booking released');tick();return}}}
async function tick(){const s=await (await fetch('/api/state')).json();rate=s.rate;hasW=s.balance!==undefined;const cl=s.claims||[];
$('st').textContent=s.status;$('info').textContent='Video time '+s.vt+'s \u00B7 '+s.pricing;
const all=s.rows.flat(),n=k=>all.filter(c=>c.state==k).length;$('sf').textContent=n('free');$('so').textContent=n('occupied');$('sb').textContent=n('booked');if($('sr'))$('sr').textContent='\u20B9'+(s.revenue||0);$('sp').textContent=all.length?Math.round(100*(n('occupied')+n('booked'))/all.length)+'%':'-';if($('au'))$('au').textContent=s.auto?'on':'off';
const cols=Math.max(1,...s.rows.map(r=>r.length));
$('grid').className='';$('grid').innerHTML=s.rows.map(r=>`<div class=row style="grid-template-columns:repeat(${cols},1fr)">`+r.map(c=>`<div class="slot ${c.state}${c.wrong?' wr':''}"><span class=ic>${c.state=='free'?'🟢':c.state=='booked'?'🕒':'🚗'}</span><b>${c.label}</b><small>${c.state}</small>${c.who?`<div class=who>${esc(c.who.name)}${c.who.plate?`<br><b>${esc(c.who.plate)}</b><br>${esc(c.who.phone)}<br>${esc(c.who.colour||'')}`:''}</div>`:''}${c.wrong?'<div class=who style="color:#fb7185"><b>&#9888; WRONG PARKING</b></div>':''}${c.state=='free'?`<button onclick="open_(${c.id},'${c.label}')">Book slot</button>`:''}${hasW&&c.state!='booked'?(cl.includes(c.id)?`<button class=g onclick="claim(${c.id},1)">Wallet billing on (turn off)</button>`:`<button class=g onclick="claim(${c.id})">Bill my wallet</button>`):''}</div>`).join('')+'</div>').join('')||'Waiting for parked cars to be detected...';
if($('ch'))$('ch').innerHTML='<tr><th>#<th>Slot<th>Plate<th>Sec<th>Amount<th>Status<th></tr>'+s.charges.map(c=>`<tr><td>${c.id}<td>${c.slot}<td>${c.plate||'-'}<td>${c.seconds}<td>\u20B9${c.amount}<td><span class="pill ${c.status}">${c.status}</span><td><a href="/receipt/${c.id}" target=_blank style=color:#67e8f9>receipt</a>${c.status=='FAILED'?` <button onclick="retry(${c.id})">Retry</button>`:''}</tr>`).join('');
$('mine').innerHTML=s.mine.map(b=>`Slot ${b.slot}: ${new Date(b.start*1e3).toLocaleTimeString()} - ${new Date(b.end*1e3).toLocaleTimeString()} (\u20B9${b.amount}) <button onclick="editB(${b.id})">Edit</button> <button class=g onclick="cancelB(${b.id})">Cancel</button>`).join('<br>')||'No upcoming bookings';
if(s.balance!==undefined)$('bal').innerHTML=' &middot; Wallet \u20B9'+(+s.balance).toFixed(2)+' <button onclick="topup()">Top up</button>'}
async function api(u,body,method){const r=await fetch(u,{method:method||'POST',headers:H,body:JSON.stringify(body||{})});const j=await r.json().catch(()=>({}));if(!r.ok)toast(j.error||'failed');return j}
async function cancelB(id){if(!confirm('Cancel this booking?'))return;const j=await api('/api/booking/'+id+'/cancel');if(j.ok){toast(j.refunded?'Cancelled, refunded \u20B9'+j.refunded:'Booking cancelled');tick();loadAdmin()}}
async function editB(id){const sm=prompt('Start in how many minutes? (0 = now)','0');if(sm===null)return;const d=prompt('Duration in seconds?','60');if(d===null)return;const j=await api('/api/booking/'+id,{start_min:sm,seconds:d},'PUT');if(j.ok){toast('Booking updated');tick()}}
async function retry(id){const j=await api('/api/retry/'+id);if(j.ok)toast('Retrying payment...');tick()}
function topup(){location.href='/payments'}
async function claim(id,off){const j=await api('/api/claim',{slot:id,off:!!off});if(j.ok){toast(off?'Wallet billing turned off':'Your wallet will be charged automatically when you leave');tick()}}
async function heat(){const h=await (await fetch('/api/history')).json();$('hm').innerHTML=h.hours.map((v,i)=>`<div title="${i}:00 - ${v==null?'no data':v+'% busy'}" style="height:34px;border-radius:4px;background:${v==null?'#ffffff10':'rgba(52,211,153,'+(0.15+v/130)+')'}"></div>`).join('');$('hmi').textContent='Hours 0-23 (local time). Average stay: '+h.avg_stay+'s over '+h.stays+' charges'+(h.busiest!=null?'. Busiest hour: '+h.busiest+':00':'')}
const esc=s=>String(s==null?'':s).replace(/[&<>"']/g,c=>'&#'+c.charCodeAt(0)+';');
async function loadAlerts(){if(!$('al'))return;const j=await (await fetch('/api/alerts')).json();
$('al').innerHTML='<tr><th>#<th>When<th>Slot<th>Parked car<th>Booked for<th>Guard call<th></tr>'+(j.alerts.map(a=>`<tr><td>${a.id}<td>${new Date(a.ts*1e3).toLocaleTimeString()}<td>${esc(a.label)}<td>${esc(a.plate)}<td>${esc(a.booked_by)} (${esc(a.expected_plate)})<td>${esc(a.status)}<td>${String(a.status).startsWith('RESOLVED')?'':`<button class=g onclick="resolveA(${a.id})">Resolve</button>`}</tr>`).join('')||'<tr><td colspan=7>No alerts</tr>')}
async function resolveA(id){await api('/api/alerts/'+id+'/resolve');loadAlerts()}
let PK=[],pid_=null;
async function loadAdmin(){if(!$('ab'))return;
const b=await (await fetch('/api/admin/bookings')).json(),now=Date.now()/1e3;
$('ab').innerHTML='<tr><th>#<th>User<th>Slot<th>From<th>To<th>Amount<th>Status<th></tr>'+(b.bookings.map(x=>{const st=x.end<now?'finished':x.start<=now?'active':'upcoming';
return `<tr><td>${x.id}<td>${esc(x.name)}<br><small>${esc(x.email)}</small><td>${esc(x.label)}<td>${new Date(x.start*1e3).toLocaleString()}<td>${new Date(x.end*1e3).toLocaleString()}<td>\u20B9${x.amount}<td>${st}<td>${st=='finished'?'':`<button class=g onclick="cancelB(${x.id})">Cancel</button>`}</tr>`}).join('')||'<tr><td colspan=8>No bookings yet</tr>');
const p=await (await fetch('/api/parkings')).json();PK=p.parkings;
$('pdi').textContent=p.discover||'';
$('pk').innerHTML='<tr><th>Name<th>Location<th>\u20B9/sec<th></tr>'+PK.map((q,i)=>`<tr><td>${esc(q.name)}${q.live?' <span class="pill PAID">live</span>':''}${q.status=='review'?' <span class="pill PENDING">review</span>':''}<td>${q.lat.toFixed(4)}, ${q.lng.toFixed(4)}<td>${+q.rate.toFixed(2)}<td>${q.status=='review'?`<button onclick="pOk(${q.id})">Approve</button> `:''}<button class=g onclick="pEdit(${i})">Edit</button>${q.live?'':` <button class=g onclick="pDel(${q.id})">Delete</button>`}</tr>`).join('')}
function pEdit(i){const p=PK[i];pid_=p.id;$('pn').value=p.name;$('pa').value=p.address||'';$('pla').value=p.lat;$('plo').value=p.lng;$('prt').value=p.live?'':p.rate;$('psl').textContent='Save changes'}
function pClear(){pid_=null;['pn','pa','pla','plo','prt'].forEach(i=>$(i).value='');$('psl').textContent='Add parking'}
function pHere(){if(!navigator.geolocation)return toast('Location unavailable');navigator.geolocation.getCurrentPosition(p=>{$('pla').value=p.coords.latitude.toFixed(6);$('plo').value=p.coords.longitude.toFixed(6)},()=>toast('Location unavailable'))}
async function pSave(){const body={name:$('pn').value,address:$('pa').value,lat:$('pla').value,lng:$('plo').value,rate:$('prt').value};const j=pid_?await api('/api/parkings/'+pid_,body,'PUT'):await api('/api/parkings',body);if(j.ok){toast('Saved');pClear();loadAdmin()}}
async function pDel(id){if(confirm('Delete this parking?')){const j=await api('/api/parkings/'+id,{},'DELETE');if(j.ok)loadAdmin()}}
async function pFind(){const j=await api('/api/parkings/discover');if(j.ok){toast('TinyFish is searching - results appear below for review');loadAdmin()}}
async function pOk(id){const j=await api('/api/parkings/'+id+'/approve');if(j.ok)loadAdmin()}
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
const cam=$('cam');function nf(){cam.onload=()=>setTimeout(nf,40);cam.onerror=()=>setTimeout(nf,500);cam.src='/frame.jpg?lot='+LOT+'&t='+Date.now()}nf();
setInterval(tick,1000);tick();heat();setInterval(heat,60000);loadAdmin();setInterval(loadAdmin,5000);loadAlerts();setInterval(loadAlerts,5000)</script>"""

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
    lines = [("Receipt #", ch["id"]), ("Slot", ch["slot"]), ("Plate", ch["plate"] or "-"), ("Vehicle", ch["vtype"] or "car"), ("Paid via", ch["method"] or "-"), ("Duration", f"{ch['seconds']} s"),
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
<header><a class=logo href=/nearby>&#128663; Smart<span>Parking</span></a><div class=user>{{name}}{% if is_admin %} (admin){% endif %}<a href=/bookings>My bookings</a><a href=/payments>My payments</a>{% if is_admin %}<a href=/admin>Admin</a><a href=/admin/payments>Payments</a>{% endif %}<a href=/logout>Logout</a></div></header>
<main style="max-width:760px;margin:0 auto;padding:20px 16px"><h2 style=margin:0>Parkings near you</h2><div class=info id=loc>Finding your location...</div>
<div style="margin-top:10px;display:flex;gap:8px"><button id=s1 onclick="srt('km')">Nearest</button><button id=s2 onclick="srt('rate')" style="opacity:.5">Cheapest</button></div><div id=list></div><button id=rl onclick=go() style="display:none;margin-top:14px">Use my location</button></main>
<script>const $=i=>document.getElementById(i),esc=s=>String(s==null?'':s).replace(/[&<>"']/g,c=>'&#'+c.charCodeAt(0)+';');
let D=null,mode='km';
function srt(m){mode=m;$('s1').style.opacity=m=='km'?1:.5;$('s2').style.opacity=m=='rate'?1:.5;draw()}
function draw(){if(!D)return;const a=D.parkings.slice();if(mode=='rate')a.sort((x,y)=>x.rate-y.rate||(x.km??1e9)-(y.km??1e9));
$('list').innerHTML=a.map((p,i)=>`<a class=pk href="/parking/${p.id}"><div><b>${esc(p.name)}</b>${i==0&&D.located&&mode=='km'?' <span class=tag>Nearest</span>':''}${i==0&&mode=='rate'?' <span class=tag>Cheapest</span>':''}<div class=sub>${esc(p.address||'')}</div><div class=sub>\u20B9${+p.rate.toFixed(2)} per second</div></div><div class=r>${p.km!=null?'<b>'+p.km+' km</b>':''}<div class=sub>${p.live?`<span class=live2>&#9679; Live video</span> &middot; ${p.free} of ${p.total} slots free`:'No live camera (demo)'}</div></div></a>`).join('')||'<div class=info>No parkings yet.</div>'}
async function load(lat,lng){D=await (await fetch('/api/parkings'+(lat==null?'':'?lat='+lat+'&lng='+lng))).json();draw()}
function go(){$('rl').style.display='none';if(!navigator.geolocation){$('loc').textContent='Location is not supported by this browser - showing all parkings.';load();return}
$('loc').textContent='Finding your location...';
navigator.geolocation.getCurrentPosition(p=>{$('loc').textContent='Sorted by distance from your current location.';load(p.coords.latitude,p.coords.longitude)},
()=>{$('loc').textContent='Could not get your location (allow location access; browsers need https or localhost) - showing all parkings.';$('rl').style.display='inline-block';load()},{timeout:10000,maximumAge:60000})}
go();setInterval(()=>{if($('list').children.length)go()},60000)</script>"""

NOLIVE = """<!doctype html><meta name=viewport content="width=device-width,initial-scale=1"><title>{{p.name}}</title>
<style>""" + THEME + """</style><header><a class=logo href=/nearby>&#128663; Smart<span>Parking</span></a><div class=user><a href=/nearby>Nearby parkings</a><a href=/logout>Logout</a></div></header>
<main style="max-width:560px;margin:40px auto;padding:0 16px"><h2>{{p.name}}</h2><p class=info>{{p.address}}</p>
<p>This parking has no live camera yet, so slots can't be shown or booked here.</p><p><a href=/nearby style=color:#67e8f9>&larr; Back to nearby parkings</a></p></main>"""

# ---------------- payments: live meter, ledger, per-plate view, wallet + Razorpay, test driver ----------------
from html import escape as html_escape
TEST_PHONE = "9000000001"
MINE = "(ch.user_id=? OR ch.plate IN (SELECT plate FROM vehicles WHERE user_id=?))"      # charges that belong to one driver (by account or by their car's plate)
DUE = "ch.booking_id IS NULL AND ch.status IN ('PENDING','FAILED')"                       # charges a driver can still pay (booking charges follow the booking flow)

def _midnight(): return time.mktime(time.localtime()[:3] + (0, 0, 0, 0, 0, -1))
def _int(v, d):
    try: return max(1, min(int(v), 1000))
    except (TypeError, ValueError): return d

def plate_owner(plate):
    """Account that owns this number plate (registered driver or the test driver), or None."""
    if not plate: return None
    with db() as c: r = c.execute("SELECT user_id FROM vehicles WHERE plate=?", (plate,)).fetchone()
    return r["user_id"] if r else None

def ensure_test_user(plate=None, balance=None):
    """A ready-made driver (TEST_EMAIL / TEST_PASSWORD) who owns one car and has money in the wallet. Pass plate to move the test
    driver to any other car. Only created once an admin exists, so the first account that registers still becomes the admin."""
    with db() as c:
        if not c.execute("SELECT 1 FROM users WHERE role='admin'").fetchone(): return None
        u = c.execute("SELECT * FROM users WHERE email=?", (TEST_EMAIL,)).fetchone()
        if plate:
            plate = norm_plate(plate)
            if not plate: raise ValueError("That is not a valid number plate")
        else: plate = u["plate"] if u and u["plate"] else None
        if not plate:                                              # default: the 2nd car of the fake pool = what slot 1 of lot1 gets
            pool = [r[0] for r in c.execute("SELECT plate FROM vehicles WHERE fake=1 ORDER BY id")]
            plate = pool[1 % len(pool)] if pool else "DL1CTEST0001"
        own = c.execute("SELECT user_id FROM vehicles WHERE plate=?", (plate,)).fetchone()
        if own and own[0] and (not u or own[0] != u["id"]): raise ValueError("That car already belongs to another driver")
        if u is None:
            uid = c.execute("INSERT INTO users(name,email,pw,role,balance,phone,plate) VALUES(?,?,?,?,?,?,?)",
                            ("Test Driver", TEST_EMAIL, generate_password_hash(TEST_PW), "user", TEST_BALANCE if balance is None else balance, TEST_PHONE, plate)).lastrowid
        else:
            uid = u["id"]
            if balance is None: c.execute("UPDATE users SET plate=? WHERE id=?", (plate, uid))
            else:
                old = u["balance"] or 0
                c.execute("UPDATE users SET plate=?, balance=? WHERE id=?", (plate, balance, uid))
                wtx(c, uid, "ADJUST", balance - old, "admin set balance")
        c.execute("UPDATE vehicles SET user_id=NULL WHERE user_id=? AND plate<>?", (uid, plate))
        if not c.execute("UPDATE vehicles SET user_id=?, owner='Test Driver', phone=?, email=? WHERE plate=?", (uid, TEST_PHONE, TEST_EMAIL, plate)).rowcount:
            c.execute("INSERT INTO vehicles(plate,owner,phone,email,fake,user_id) VALUES(?,?,?,?,1,?)", (plate, "Test Driver", TEST_PHONE, TEST_EMAIL, uid))
    return plate

def live_sessions():
    """Every car standing in a slot right now: plate, slot, time parked (video seconds - the clock billing uses) and the charge so far."""
    out = []
    for k in LOTS:
        with using(k), LOCK:
            labs = {s: slot_label(i, j) for i, r in enumerate(layout()[1]) for j, s in enumerate(r)}
            for s, on in list(S["occ"].items()):
                if not on: continue
                secs, vt = max(0, S["vt"] - S["since"].get(s, S["vt"])), S["vtype"].get(s, "car")
                out.append({"lot": k, "slot": s, "label": labs.get(s, f"#{s}"), "plate": S["who"].get(s), "vtype": vt, "seconds": int(secs), "amount": price(secs, vt, s)})
    return out

@app.route("/api/admin/payments/live")
@need_admin
def pay_live():
    L = live_sessions(); V = vehicle_info([s["plate"] for s in L])
    for s in L: v = V.get(s["plate"]) or {}; s["owner"], s["phone"] = v.get("owner"), v.get("phone")
    return jsonify(live=L, accrued=sum(s["amount"] for s in L))

@app.route("/api/admin/payments/ledger")
@need_admin
def pay_ledger():
    """Ledger = every charge, newest first, with who/what/how it was paid. Filters: status, plate (exact=1 for one car)."""
    a, w, p = request.args, [], []
    if a.get("status") in ("PAID", "PENDING", "FAILED"): w.append("ch.status=?"); p.append(a["status"])
    q = re.sub(r"[^A-Z0-9]", "", (a.get("plate") or "").upper())
    if q == "UNKNOWN": w.append("ch.plate IS NULL")
    elif q and a.get("exact"): w.append("ch.plate=?"); p.append(q)
    elif q: w.append("ch.plate LIKE ?"); p.append(f"%{q}%")
    lab = slot_labels()
    with db() as c:
        rows = [dict(r) for r in c.execute("SELECT ch.id,ch.created,ch.paid_at,ch.lot,ch.slot,ch.plate,ch.vtype,ch.seconds,ch.amount,ch.status,ch.method,ch.run,u.name payer,v.owner,v.phone "
                                           "FROM charges ch LEFT JOIN users u ON u.id=ch.user_id LEFT JOIN vehicles v ON v.plate=ch.plate"
                                           + (" WHERE " + " AND ".join(w) if w else "") + " ORDER BY ch.id DESC LIMIT ?", (*p, _int(a.get("limit"), 200)))]
        t = dict(c.execute("SELECT COALESCE(SUM(CASE WHEN status='PAID' THEN amount END),0) paid, COALESCE(SUM(CASE WHEN status='PENDING' THEN amount END),0) pending, "
                           "COALESCE(SUM(CASE WHEN status='FAILED' THEN amount END),0) failed, COUNT(*) n FROM charges").fetchone())
        t["today"] = c.execute("SELECT COALESCE(SUM(amount),0) FROM charges WHERE status='PAID' AND COALESCE(paid_at,created)>=?", (_midnight(),)).fetchone()[0]
        t["plans"] = c.execute("SELECT COALESCE(SUM(price),0) FROM user_plans").fetchone()[0]
    for r in rows: r["label"] = lab.get(r["slot"], f"#{r['slot']}")
    return jsonify(rows=rows, totals=t)

@app.route("/api/admin/payments/by_plate")
@need_admin
def pay_by_plate():
    """Same money, grouped per number plate: stays, time, billed / paid / pending, wallet, plus the car's running meter if it is parked now."""
    with db() as c:
        rows = [dict(r) for r in c.execute(
            "SELECT COALESCE(ch.plate,'UNKNOWN') plate, COUNT(*) stays, SUM(ch.seconds) seconds, SUM(ch.amount) billed, "
            "SUM(CASE WHEN ch.status='PAID' THEN ch.amount ELSE 0 END) paid, SUM(CASE WHEN ch.status='PENDING' THEN ch.amount ELSE 0 END) pending, "
            "SUM(CASE WHEN ch.status='FAILED' THEN ch.amount ELSE 0 END) failed, MAX(ch.created) last_stay, v.owner, v.phone, u.balance wallet "
            "FROM charges ch LEFT JOIN vehicles v ON v.plate=ch.plate LEFT JOIN users u ON u.id=v.user_id GROUP BY COALESCE(ch.plate,'UNKNOWN') ORDER BY last_stay DESC")]
    live = {s["plate"]: s for s in live_sessions() if s["plate"]}
    for r in rows: r["live"] = live.pop(r["plate"], None)
    V = vehicle_info(list(live))
    for pl_, s in live.items():                                    # parked right now, nothing billed yet
        v = V.get(pl_) or {}
        rows.insert(0, {"plate": pl_, "stays": 0, "seconds": 0, "billed": 0, "paid": 0, "pending": 0, "failed": 0, "last_stay": None, "owner": v.get("owner"), "phone": v.get("phone"), "wallet": None, "live": s})
    return jsonify(rows=rows)

def video_seconds(lot):
    """Total length of the lot's videos in seconds (0 if unreadable)."""
    tot = 0.0
    for f in lot_videos(lot):
        try:
            cap = cv2.VideoCapture(f); fps, n = cap.get(cv2.CAP_PROP_FPS) or 0, cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0; cap.release()
            if fps > 0: tot += n / fps
        except Exception: pass
    return tot

@app.route("/api/admin/payments/check")
@need_admin
def pay_check():
    """Sanity check of the billing data per lot: do plates look right, are durations possible for the video, anything odd?"""
    out = []
    with db() as c:
        for k in LOTS:
            vs = video_seconds(k); rows = c.execute("SELECT * FROM charges WHERE COALESCE(lot,?)=?", (FIRST, k)).fetchall()
            known = {r[0] for r in c.execute("SELECT plate FROM vehicles")}
            issues = []
            noplate = [r["id"] for r in rows if not r["plate"]]
            unknown = sorted({r["plate"] for r in rows if r["plate"] and r["plate"] not in known})
            toolong = [r["id"] for r in rows if vs and r["seconds"] > vs + 1]
            zero = [r["id"] for r in rows if (r["seconds"] or 0) <= 0]
            if not lot_videos(k): issues.append("no video found for this lot")
            if noplate: issues.append(f"{len(noplate)} charge(s) have no plate (ids {noplate[:8]})")
            if unknown: issues.append(f"plates not in the vehicles table: {unknown[:8]}")
            if toolong: issues.append(f"{len(toolong)} charge(s) longer than the video ({int(vs)} s): ids {toolong[:8]}")
            if zero: issues.append(f"{len(zero)} charge(s) with 0 seconds")
            out.append({"lot": k, "dual": k in DUAL, "plate_source": "plate reader (gate camera)" if k in DUAL else "fake pool (no plate reader)", "video_seconds": int(vs), "charges": len(rows),
                        "avg_seconds": int(sum(r["seconds"] or 0 for r in rows) / len(rows)) if rows else 0, "longest": max((r["seconds"] or 0 for r in rows), default=0), "issues": issues})
    return jsonify(lots=out)

@app.route("/api/admin/visits")
@need_admin
def visits_api():
    """The permanent in / out log: one row per visit (lot, slot, plate, in, out, parked seconds, which video loop). Filters: plate, lot."""
    a, w, p = request.args, [], []
    q = re.sub(r"[^A-Z0-9]", "", (a.get("plate") or "").upper())
    if q: w.append("v.plate LIKE ?"); p.append(f"%{q}%")
    if a.get("lot") in LOTS: w.append("v.lot=?"); p.append(a["lot"])
    with db() as c:
        rows = [dict(r) for r in c.execute("SELECT v.*, ve.owner, ve.phone FROM visits v LEFT JOIN vehicles ve ON ve.plate=v.plate" + (" WHERE " + " AND ".join(w) if w else "") + " ORDER BY v.id DESC LIMIT ?", (*p, _int(a.get("limit"), 300)))]
        n = c.execute("SELECT COUNT(*) FROM visits").fetchone()[0]
    labs = {}
    for r in rows:
        if r["lot"] not in labs:
            with using(r["lot"] if r["lot"] in LOTS else FIRST): labs[r["lot"]] = slot_labels()
        r["label"] = labs[r["lot"]].get(r["slot"], f"#{r['slot']}")
    return jsonify(rows=rows, total=n)

@app.route("/api/admin/test_user", methods=["GET", "POST"])
@need_admin
def test_user():
    if request.method == "POST":
        d = jd()
        try: ensure_test_user(d.get("plate") or None, None if d.get("balance") in (None, "") else float(d["balance"]))
        except (ValueError, TypeError) as e: return jsonify(error=str(e) or "invalid"), 400
    ensure_test_user()
    with db() as c: u = c.execute("SELECT name,email,plate,balance FROM users WHERE email=?", (TEST_EMAIL,)).fetchone()
    return jsonify(user=dict(u) if u else None, password=TEST_PW)

# ---- driver side: my charges, pay from wallet, pay / top up with Razorpay ----
@app.route("/api/my/payments")
@need_login
def my_payments():
    uid = session["uid"]
    with db() as c:
        rows = [dict(r) for r in c.execute(f"SELECT ch.id,ch.slot,ch.plate,ch.seconds,ch.amount,ch.status,ch.method,ch.created,ch.paid_at,"
                                           f"CASE WHEN {DUE} THEN 1 ELSE 0 END due FROM charges ch WHERE {MINE} ORDER BY ch.id DESC LIMIT 100", (uid, uid))]
        u = c.execute("SELECT balance,plate FROM users WHERE id=?", (uid,)).fetchone()
        wl = [dict(r) for r in c.execute("SELECT ts,kind,amount,ref,balance FROM wallet_tx WHERE user_id=? ORDER BY id DESC LIMIT 30", (uid,))]
    return jsonify(wallet_tx=wl, charges=rows, balance=u["balance"] or 0, plate=u["plate"], rzp=RZP, wallet=WALLET, pending=sum(r["amount"] for r in rows if r["due"]))

@app.route("/api/charge/<int:cid>/wallet", methods=["POST"])
@need_login
def pay_from_wallet(cid):
    if not WALLET: return jsonify(error="wallet is disabled (set WALLET=1)"), 404
    uid = session["uid"]
    with db() as c:
        ch = c.execute(f"SELECT ch.* FROM charges ch WHERE ch.id=? AND {MINE}", (cid, uid, uid)).fetchone()
        if not ch: return jsonify(error="not found"), 404
        if not c.execute("UPDATE charges SET status='PAID', run='WALLET', method='WALLET', paid_at=?, user_id=? WHERE id=? AND booking_id IS NULL AND status IN ('PENDING','FAILED')",
                         (time.time(), uid, cid)).rowcount: return jsonify(error="Already paid or not payable here"), 409
        if not c.execute("UPDATE users SET balance=balance-? WHERE id=? AND balance>=?", (ch["amount"], uid, ch["amount"])).rowcount:
            c.rollback(); return jsonify(error="Not enough wallet balance - top up or pay with Razorpay"), 402
        wtx(c, uid, "CHARGE", -ch["amount"], f"charge #{cid}")
    paid_alert(cid)
    return jsonify(ok=True)

def rzp_api(path, body):
    r = requests.post("https://api.razorpay.com/v1/" + path, auth=(RZP_KEY, RZP_SECRET), json=body, timeout=20)
    j = r.json()
    if r.status_code >= 400: raise RuntimeError((j.get("error") or {}).get("description") or f"Razorpay error {r.status_code}")
    return j

def rzp_sig_ok(order_id, pay_id, sig):
    return hmac.compare_digest(hmac.new(RZP_SECRET.encode(), f"{order_id}|{pay_id}".encode(), "sha256").hexdigest(), sig or "")

@app.route("/api/rzp/order", methods=["POST"])
@need_login
def rzp_order():
    """Create a Razorpay order for one pending charge ({charge_id}) or a wallet top-up ({topup: rupees}). The amount is always taken from our database."""
    if not RZP: return jsonify(error="Razorpay is not set up - add RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET to .env"), 400
    d, uid = jd(), session["uid"]; cid = None; pkey = None
    with db() as c:
        if d.get("charge_id") is not None:
            try: cid = int(d["charge_id"])
            except (ValueError, TypeError): return jsonify(error="invalid"), 400
            ch = c.execute(f"SELECT ch.* FROM charges ch WHERE ch.id=? AND {MINE}", (cid, uid, uid)).fetchone()
            if not ch: return jsonify(error="not found"), 404
            if ch["booking_id"] is not None or ch["status"] not in ("PENDING", "FAILED"): return jsonify(error="Already paid or not payable here"), 409
            kind, amt = "charge", float(ch["amount"])
        elif d.get("plan") is not None:
            p = plan_by_key(d.get("plan"))
            if not p: return jsonify(error="unknown plan"), 400
            kind, amt, pkey = "plan", float(plan_fee(p)), p["key"]
        else:
            try: amt = float(d.get("topup", 0))
            except (ValueError, TypeError): return jsonify(error="invalid"), 400
            if not 1 <= amt <= 10000: return jsonify(error="amount must be between 1 and 10000"), 400
            kind = "topup"
        u = c.execute("SELECT name,email,phone FROM users WHERE id=?", (uid,)).fetchone()
    try: o = rzp_api("orders", {"amount": int(round(amt * 100)), "currency": "INR", "receipt": f"{kind[0]}{cid or uid}-{int(time.time())}",
                                "notes": {"kind": kind, "charge_id": str(cid or ""), "user_id": str(uid), "plan": pkey or ""}})
    except Exception as e: return jsonify(error="Razorpay: " + str(e)[:120]), 502
    with db() as c: c.execute("INSERT INTO rzp_orders(order_id,kind,charge_id,user_id,amount,status,created,plan) VALUES(?,?,?,?,?,'CREATED',?,?)", (o["id"], kind, cid, uid, amt, time.time(), pkey))
    return jsonify(order_id=o["id"], amount=o["amount"], key=RZP_KEY, name=u["name"], email=u["email"], phone=u["phone"])

def rzp_apply(order_id, pay_id, uid=None):
    """A Razorpay payment is confirmed (signature checked by the caller). Applied exactly once even if the browser and the webhook both report it.
    Returns the order kind, or None for an unknown order."""
    paid_cid = None
    with db() as c:
        o = c.execute("SELECT * FROM rzp_orders WHERE order_id=?", (order_id,)).fetchone()
        if not o or (uid is not None and o["user_id"] != uid): return None
        if c.execute("UPDATE rzp_orders SET status='PAID', payment_id=? WHERE order_id=? AND status<>'PAID'", (pay_id, order_id)).rowcount:
            if o["kind"] == "topup": c.execute("UPDATE users SET balance=balance+? WHERE id=?", (o["amount"], o["user_id"])); wtx(c, o["user_id"], "TOPUP", o["amount"], "Razorpay " + str(pay_id))
            elif o["kind"] == "plan": activate_plan(c, o["user_id"], o["plan"], order_id)
            elif c.execute("UPDATE charges SET status='PAID', run=?, method='RAZORPAY', paid_at=?, user_id=COALESCE(user_id,?) WHERE id=? AND status IN ('PENDING','FAILED')",
                           (pay_id, time.time(), o["user_id"], o["charge_id"])).rowcount: paid_cid = o["charge_id"]
            else: c.execute("UPDATE users SET balance=balance+? WHERE id=?", (o["amount"], o["user_id"])); wtx(c, o["user_id"], "CREDIT", o["amount"], "double payment kept as credit")      # charge was settled another way meanwhile
    if paid_cid: paid_alert(paid_cid)
    return o["kind"]

@app.route("/api/rzp/verify", methods=["POST"])
@need_login
def rzp_verify():
    d = jd(); oid, pid, sig = (str(d.get(k, "")) for k in ("razorpay_order_id", "razorpay_payment_id", "razorpay_signature"))
    if not (RZP and oid and pid and rzp_sig_ok(oid, pid, sig)): return jsonify(error="payment signature does not match"), 400
    kind = rzp_apply(oid, pid, session["uid"])
    return jsonify(ok=True, kind=kind) if kind else (jsonify(error="unknown order"), 404)

@app.route("/webhook/razorpay", methods=["POST"])       # Razorpay dashboard -> Webhooks: URL {PUBLIC_URL}/webhook/razorpay, secret = RAZORPAY_WEBHOOK_SECRET, events payment.captured + order.paid
def rzp_webhook():
    raw = request.get_data()
    if not (RZP_WEBHOOK and hmac.compare_digest(hmac.new(RZP_WEBHOOK.encode(), raw, "sha256").hexdigest(), request.headers.get("X-Razorpay-Signature", ""))): return ("forbidden", 403)
    try: ev = json.loads(raw); pay = ev["payload"]["payment"]["entity"]
    except (ValueError, KeyError, TypeError): return jsonify(ok=True)           # events without a payment are acknowledged and ignored
    if ev.get("event") in ("payment.captured", "order.paid") and pay.get("order_id"): rzp_apply(pay["order_id"], pay.get("id"))
    return jsonify(ok=True)

# ================= Smart parking via machine learning =================
# Every entry and exit is saved in the `visits` table (lot, slot, plate, in, out, seconds) - even free stays, and again on every replay of the video, never deleted. A small model is trained on each driver's own history:
# a recency-weighted ridge regression (scikit-learn) of parked seconds per day (trend + day-of-week pattern; a flat average while there is under a week of data).
# The learned weights are saved in the `usage_model` table, so the model survives restarts, is retrained when a new stay arrives, and forecasts the next days.
# The forecast is then priced against every plan to find the cheapest option.
import datetime as _dt
import numpy as _np
try: from sklearn.linear_model import Ridge as _Ridge      # scikit-learn does the regression; without it the same maths is solved with NumPy
except ImportError: _Ridge = None; print("scikit-learn not installed - using the NumPy fallback (pip install scikit-learn)")

def plan_by_key(k): return next((p for p in PLANS if p["key"] == str(k)), None)
def plan_fee(p): return max(1, int(round(p["secs"] * RATE * (1 - p["off"]))))

def use_plan_secs(uid, seconds):
    """Take up to `seconds` of parked time from the driver's active plans (earliest expiry first). Returns how many seconds the plans covered."""
    got, now = 0.0, time.time()
    with db() as c:
        for p in c.execute("SELECT id,included,used FROM user_plans WHERE user_id=? AND status='ACTIVE' AND end>? AND used<included ORDER BY end", (uid, now)).fetchall():
            if got >= seconds: break
            take = min(seconds - got, p["included"] - p["used"])
            c.execute("UPDATE user_plans SET used=used+? WHERE id=?", (take, p["id"])); got += take
    return got

def activate_plan(c, uid, key, order_id=None):
    p = plan_by_key(key)
    if not p: return None
    now = time.time()
    return c.execute("INSERT INTO user_plans(user_id,plan,name,start,end,included,used,price,order_id,status) VALUES(?,?,?,?,?,?,0,?,?,'ACTIVE')",
                     (uid, p["key"], p["name"], now, now + p["days"] * 86400, p["secs"], plan_fee(p), order_id)).lastrowid

VMINE = "(v.user_id=? OR v.plate IN (SELECT plate FROM vehicles WHERE user_id=?)) AND v.status='CLOSED' AND v.seconds>0"      # closed visits of one driver (by account or by their car's plate)

def _usage_rows(c, uid):
    return c.execute(f"SELECT v.entered created, v.seconds seconds, COALESCE(ch.amount,0) amount, COALESCE(ch.plan_secs,0) ps FROM visits v LEFT JOIN charges ch ON ch.id=v.charge_id WHERE {VMINE} AND v.entered>? ORDER BY v.entered",
                     (uid, uid, time.time() - 120 * 86400)).fetchall()

def _usage_sig(c, uid):
    r = c.execute(f"SELECT COUNT(*), COALESCE(MAX(v.id),0) FROM visits v WHERE {VMINE}", (uid, uid)).fetchone()
    return f"{r[0]}:{r[1]}:{_dt.date.today().toordinal()}"

def _dow_vec(ordinal): v = [0.0] * 7; v[ordinal % 7] = 1.0; return v
def _features(ordinal, k_days_ahead): return [1.0, k_days_ahead / 30.0] + _dow_vec(ordinal)

def ml_train(uid):
    """Learn this driver's parking pattern from their saved history and store the model."""
    today = _dt.date.today().toordinal()
    with db() as c: rows = _usage_rows(c, uid); sig = _usage_sig(c, uid)
    m = {"asof": today, "n_stays": len(rows), "n_days": 0, "type": "none", "beta": [0.0], "sigma": 0.0, "rate_eff": float(RATE), "total_secs": 0, "spent": 0, "covered": 0,
         "avg_day": 0.0, "last7": 0, "daily14": [0] * 14, "avg_stay": 0.0, "trained": time.time()}
    if rows:
        by_day = {}
        for r in rows:
            d = _dt.date.fromtimestamp(r["created"]).toordinal(); by_day[d] = by_day.get(d, 0.0) + (r["seconds"] or 0)
        first = max(min(by_day), today - 59); n = today - first + 1
        y = _np.array([by_day.get(first + i, 0.0) for i in range(n)])
        w = 0.5 ** ((n - 1 - _np.arange(n)) / 14.0)
        paid = [(r["amount"], max(0, (r["seconds"] or 0) - r["ps"])) for r in rows if r["amount"] and (r["seconds"] or 0) - r["ps"] > 0]
        m.update(n_days=n, total_secs=int(sum(r["seconds"] or 0 for r in rows)), spent=int(sum(r["amount"] or 0 for r in rows)), covered=int(sum(r["ps"] for r in rows)),
                 rate_eff=(sum(a for a, _ in paid) / sum(s for _, s in paid)) if paid else float(RATE), avg_stay=sum(r["seconds"] or 0 for r in rows) / len(rows),
                 last7=int(sum(by_day.get(today - i, 0) for i in range(7))), daily14=[int(by_day.get(today - 13 + i, 0)) for i in range(14)])
        m["avg_day"] = m["total_secs"] / n
        if n < 7:                                              # too little history for a weekly pattern: recency-weighted average
            lvl = float((w * y).sum() / w.sum()); m.update(type="flat", beta=[lvl], sigma=float(_np.sqrt((w * (y - lvl) ** 2).sum() / w.sum())))
        else:                                                  # weighted ridge regression: level + trend + day-of-week
            X = _np.array([_features(first + i, i - (n - 1)) for i in range(n)])
            try:
                if _Ridge is not None:                         # scikit-learn: columns 1.. are trend + 7 weekday flags, the intercept is the level (not penalised)
                    reg = _Ridge(alpha=2.0, fit_intercept=True).fit(X[:, 1:], y, sample_weight=w)
                    beta = _np.concatenate([[reg.intercept_], reg.coef_]); res = y - reg.predict(X[:, 1:])
                else:
                    R = _np.eye(X.shape[1]) * 2.0; R[0, 0] = 0
                    beta = _np.linalg.solve(X.T @ (w[:, None] * X) + R, X.T @ (w * y)); res = y - X @ beta
                m.update(type="ridge", lib="scikit-learn" if _Ridge is not None else "numpy", beta=[float(b) for b in beta], sigma=float(_np.sqrt((w * res ** 2).sum() / w.sum())))
            except _np.linalg.LinAlgError:
                lvl = float((w * y).sum() / w.sum()); m.update(type="flat", beta=[lvl], sigma=float(y.std()))
    with db() as c: c.execute("INSERT OR REPLACE INTO usage_model(user_id,trained,sig,data) VALUES(?,?,?,?)", (uid, time.time(), sig, json.dumps(m)))
    return m

def get_model(uid):
    """The saved model for this driver; retrained only when new stays arrived (or on a new day)."""
    with db() as c:
        r = c.execute("SELECT sig,data FROM usage_model WHERE user_id=?", (uid,)).fetchone()
        sig = _usage_sig(c, uid)
    if r and r["sig"] == sig:
        try: return json.loads(r["data"])
        except ValueError: pass
    return ml_train(uid)

def ml_forecast(m, days):
    """Expected parked seconds over the next `days` days, with a low / high range (about an 80% band)."""
    if m["type"] == "none": return 0.0, 0.0, 0.0
    tot = 0.0
    for k in range(1, days + 1):
        tot += max(0.0, m["beta"][0]) if m["type"] == "flat" else max(0.0, float(_np.dot(m["beta"], _features(m["asof"] + k, k))))
    band = 1.28 * m["sigma"] * (days ** 0.5)
    return tot, max(0.0, tot - band), tot + band

def ml_insights(uid):
    m = get_model(uid); r = m["rate_eff"]; plans = []
    for p in PLANS:
        S, lo, hi = ml_forecast(m, p["days"]); fee = plan_fee(p)
        cost = lambda secs: fee + r * max(0.0, secs - p["secs"])
        plans.append({"key": p["key"], "name": p["name"], "days": p["days"], "secs": p["secs"], "off": int(round(p["off"] * 100)), "fee": fee, "breakeven": fee / r if r else 0,
                      "expected_secs": round(S), "no_plan": round(r * S), "with_plan": round(cost(S)), "save": round(r * S - cost(S)),
                      "save_low": round(r * lo - cost(lo)), "save_high": round(r * hi - cost(hi))})
    good = [p for p in plans if p["save"] > 0]
    best = max(good, key=lambda p: p["save"] / p["days"])["key"] if good and m["n_stays"] else None
    for p in plans: p["best"] = p["key"] == best
    S30, lo30, hi30 = ml_forecast(m, 30)
    if not m["n_stays"]: msg = "I have not seen you park yet. Park once and I start learning your pattern; the plans below show what each one costs."
    elif best:
        b = next(p for p in plans if p["key"] == best)
        msg = f"Based on your {m['n_stays']} stay(s) I expect about {round(S30 / 60, 1)} min of parking in the next 30 days. The {b['name']} would cost {b['with_plan']} instead of about {b['no_plan']} pay-as-you-go, saving you about {b['save']}."
    else: msg = f"Based on your {m['n_stays']} stay(s) I expect about {round(S30 / 60, 1)} min of parking in the next 30 days. At that level pay-as-you-go is still the cheapest, so no plan is worth buying yet."
    conf = "Still learning" if m["n_stays"] < 3 else ("Early estimate" if m["n_days"] < 7 else "Good confidence")
    now = time.time()
    with db() as c: act = [dict(x) for x in c.execute("SELECT name,end,included,used FROM user_plans WHERE user_id=? AND status='ACTIVE' AND end>? ORDER BY end", (uid, now))]
    for a in act: a["left"] = max(0, a["included"] - a["used"])
    return {"message": msg, "confidence": conf, "model": m["type"], "usage": {k: m[k] for k in ("n_stays", "n_days", "total_secs", "spent", "covered", "avg_day", "avg_stay", "last7", "daily14", "rate_eff")},
            "forecast": {"days": 30, "secs": round(S30), "low": round(lo30), "high": round(hi30), "cost": round(r * S30)}, "plans": plans, "active": act, "trained": m["trained"]}

@app.route("/api/my/plans")
@need_login
def my_plans():
    d = ml_insights(session["uid"]); d.update(rzp=RZP, wallet=WALLET)
    return jsonify(d)

@app.route("/api/plan/wallet", methods=["POST"])
@need_login
def plan_wallet():
    if not WALLET: return jsonify(error="wallet is disabled (set WALLET=1)"), 404
    p = plan_by_key(jd().get("plan")); uid = session["uid"]
    if not p: return jsonify(error="unknown plan"), 400
    fee = plan_fee(p)
    with db() as c:
        if not c.execute("UPDATE users SET balance=balance-? WHERE id=? AND balance>=?", (fee, uid, fee)).rowcount: return jsonify(error="Not enough wallet balance - top up or pay with Razorpay"), 402
        wtx(c, uid, "PLAN", -fee, p["name"]); activate_plan(c, uid, p["key"])
    return jsonify(ok=True)

# ---- pages ----
PAY_CSS = r"""main{max-width:1180px;margin:0 auto;padding:18px 20px 48px}h2{margin:0 0 4px;font-size:22px}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:16px 0}
.stat{background:#ffffff0d;border:1px solid #ffffff1a;border-radius:14px;padding:12px 16px}.stat small{color:#94a3b8}.stat b{display:block;font-size:24px;margin-top:2px}
.card{background:#ffffff0d;border:1px solid #ffffff1a;border-radius:16px;padding:16px;margin-top:16px}.card h3{margin:0 0 10px;font-size:15px;color:#cbd5e1;font-weight:600}
.sc{overflow-x:auto}table{width:100%;font-size:13px;border-collapse:collapse}th{color:#94a3b8;font-weight:500;text-align:left;padding:6px;white-space:nowrap}td{padding:8px 6px;border-top:1px solid #ffffff1a}.n{text-align:right;font-variant-numeric:tabular-nums}
.plate{font:700 13px ui-monospace,Menlo,Consolas,monospace;letter-spacing:.06em;background:#fde047;color:#1c1917;padding:2px 8px;border-radius:5px;border:1px solid #0008;white-space:nowrap}
.pill{padding:2px 9px;border-radius:99px;font-size:11px;font-weight:700;background:#ffffff1a}.PAID{background:#10b98126;color:#6ee7b7}.PENDING{background:#f59e0b26;color:#fcd34d}.FAILED{background:#ef444426;color:#fca5a5}
button{padding:7px 12px;border:0;border-radius:8px;background:linear-gradient(135deg,#34d399,#06b6d4);color:#04210f;font-weight:700;cursor:pointer;font-size:12px}button.off{background:#ffffff14;color:#cbd5e1}
button:focus-visible,input:focus-visible,select:focus-visible{outline:2px solid #67e8f9;outline-offset:2px}
input,select{padding:8px 10px;border-radius:9px;border:1px solid #ffffff1a;background:#0b1020;color:#fff;font-size:13px}.row{display:flex;gap:8px;flex-wrap:wrap;align-items:center}
#t{position:fixed;bottom:20px;right:20px;background:#10b981;color:#04210f;padding:12px 18px;border-radius:12px;font-weight:700;display:none;z-index:10}
.pg{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;margin-top:12px}.pc{border:1px solid #ffffff1a;border-radius:14px;padding:14px;background:#ffffff08}.pc.best{border-color:#34d399;box-shadow:0 0 0 1px #34d399}
.bars{display:flex;gap:3px;align-items:flex-end;height:48px;margin:8px 0}.bars i{flex:1;background:linear-gradient(#34d399,#06b6d4);border-radius:3px 3px 0 0;min-height:2px}"""
PAY_JS = r"""const $=i=>document.getElementById(i),esc=s=>String(s==null?'':s).replace(/[&<>"']/g,c=>'&#'+c.charCodeAt(0)+';');
const H={'Content-Type':'application/json','X-CSRF-Token':'__CSRF__'};
const rs=n=>'\u20B9'+Math.round(+n||0),pad=n=>String(n).padStart(2,'0');
const dur=s=>{s=Math.round(s||0);const h=Math.floor(s/3600),m=Math.floor(s%3600/60);return(h?h+':'+pad(m):m)+':'+pad(s%60)};
const pl=p=>!p||p=='UNKNOWN'?'<span class=pill>unknown</span>':`<span class=plate>${esc(p)}</span>`;
const when=t=>t?new Date(t*1e3).toLocaleString([],{day:'2-digit',month:'short',hour:'2-digit',minute:'2-digit'}):'-';
function toast(x){const t=$('t');t.textContent=x;t.style.display='block';setTimeout(()=>t.style.display='none',3000)}
async function api(u,b,m){const r=await fetch(u,{method:m||'POST',headers:H,body:JSON.stringify(b||{})});const j=await r.json().catch(()=>({}));if(!r.ok)toast(j.error||'failed');return j}"""

PAY_ADMIN = """<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1"><title>Payments - Smart Parking</title><style>""" + THEME + PAY_CSS + """</style>
<header><a class=logo href=/nearby>&#128663; Smart<span>Parking</span></a><div class=user>__NAME__ (admin)<a href=/admin>Admin</a><a href=/admin/payments>Payments</a><a href=/nearby>Nearby parkings</a><a href=/logout>Logout</a></div></header>
<main><h2>Payments</h2><div class=info>Cars show up here the moment they park. The charge is booked when the car leaves its slot.</div>
<div class=stats><div class=stat><small>Collected</small><b id=sp style=color:#6ee7b7>-</b></div><div class=stat><small>Collected today</small><b id=st>-</b></div><div class=stat><small>Plan sales</small><b id=sps>-</b></div>
<div class=stat><small>Pending</small><b id=sd style=color:#fcd34d>-</b></div><div class=stat><small>Failed</small><b id=sf style=color:#fca5a5>-</b></div>
<div class=stat><small>Running now (<span id=sn>0</span> parked)</small><b id=sl>-</b></div></div>
<div class=card><h3>Parked now</h3><div class=sc><table id=live></table></div></div>
<div class=card><div class=row style="justify-content:space-between"><div class=row><button id=b-ledger onclick="sw('ledger')">Ledger</button><button id=b-plate class=off onclick="sw('plate')">By number plate</button><button id=b-wallet class=off onclick="sw('wallet')">Wallet history</button><button id=b-visits class=off onclick="sw('visits')">In / Out log</button></div>
<div class=row><select id=fs onchange=ledger() aria-label="Status"><option value="">All statuses</option><option>PAID</option><option>PENDING</option><option>FAILED</option></select>
<input id=fq placeholder="Search plate" oninput="fp='';ledger()"><a href="/admin/export.csv" style=color:#67e8f9;font-size:13px>Ledger CSV</a><a href="/admin/export.csv?kind=plates" style=color:#67e8f9;font-size:13px>Per-plate CSV</a><a href="/admin/export.csv?kind=wallet" style=color:#67e8f9;font-size:13px>Wallet CSV</a><a href="/admin/export.csv?kind=visits" style=color:#67e8f9;font-size:13px>In/Out CSV</a></div></div>
<div id=chip class=info></div><div id=v-ledger class=sc style=margin-top:10px><table id=led></table></div><div id=v-plate class=sc style="margin-top:10px;display:none"><table id=bp></table></div><div id=v-wallet class=sc style="margin-top:10px;display:none"><table id=wl></table></div><div id=v-visits class=sc style="margin-top:10px;display:none"><table id=vl></table></div></div>
<div class=card><h3>Data check (plates and durations per lot)</h3><div class=info id=ck>-</div><button class=off onclick=check() style=margin-top:8px>Run check</button></div>
<div class=card><h3>Test driver</h3><div class=info id=tu></div><div class=row style=margin-top:10px><input id=tp placeholder="Car plate (blank = keep)" style=text-transform:uppercase><input id=tb type=number min=0 placeholder="Wallet balance"><button onclick=saveTU()>Create / update test driver</button></div></div></main><div id=t></div>
<script>""" + PAY_JS + r"""
let tab='ledger',fp='';
function sw(t){tab=t;for(const k of ['ledger','plate','wallet','visits']){$('v-'+k).style.display=t==k?'':'none';$('b-'+k).className=t==k?'':'off'}$('fs').style.display=$('fq').style.display=t=='ledger'?'':'none'}
async function live(){const j=await (await fetch('/api/admin/payments/live')).json();$('sl').textContent=rs(j.accrued);$('sn').textContent=j.live.length;
$('live').innerHTML='<tr><th>Plate<th>Owner<th>Slot<th>Parked for<th class=n>Running charge</tr>'+(j.live.map(s=>`<tr><td>${pl(s.plate)}<td>${esc(s.owner||'-')}<br><small>${esc(s.phone||'')}</small><td>${esc(s.lot)} &middot; ${esc(s.label)}<td>${dur(s.seconds)}<td class=n>${rs(s.amount)}</tr>`).join('')||'<tr><td colspan=5 class=info>No car is parked right now.</tr>')}
async function ledger(){const q=new URLSearchParams({status:$('fs').value,plate:fp||$('fq').value,exact:fp?1:''});const j=await (await fetch('/api/admin/payments/ledger?'+q)).json(),T=j.totals;
$('sp').textContent=rs(T.paid);$('st').textContent=rs(T.today);$('sps').textContent=rs(T.plans);$('sd').textContent=rs(T.pending);$('sf').textContent=rs(T.failed);
$('chip').innerHTML=fp?`Showing stays of ${pl(fp)} <button class=off onclick="fp='';$('fq').value='';ledger()">Show all</button>`:'';
$('led').innerHTML='<tr><th>#<th>Time<th>Plate<th>Owner<th>Slot<th>Parked<th class=n>Amount<th>Status<th>Paid via<th></tr>'+(j.rows.map(c=>`<tr><td>${c.id}<td>${when(c.created)}<td>${pl(c.plate)}<td>${esc(c.payer||c.owner||'-')}<td>${esc(c.lot||'')} &middot; ${esc(c.label)}<td>${dur(c.seconds)}<td class=n>${rs(c.amount)}<td><span class="pill ${c.status}">${c.status}</span><td>${esc(c.method||'-')}<td><a href="/receipt/${c.id}" target=_blank style=color:#67e8f9>receipt</a>${c.status=='FAILED'?` <button onclick="retry(${c.id})">Retry</button>`:''}${c.status!='PAID'?` <button class=off onclick="cash(${c.id})">Cash</button>`:''}</tr>`).join('')||'<tr><td colspan=10 class=info>No payments yet.</tr>')}
async function byPlate(){const j=await (await fetch('/api/admin/payments/by_plate')).json();
$('bp').innerHTML='<tr><th>Plate<th>Owner<th class=n>Stays<th class=n>Total time<th class=n>Billed<th class=n>Paid<th class=n>Pending<th class=n>Wallet<th>Now<th></tr>'+(j.rows.map(r=>{const due=(r.pending||0)+(r.failed||0);
return `<tr><td>${pl(r.plate)}<td>${esc(r.owner||'-')}<br><small>${esc(r.phone||'')}</small><td class=n>${r.stays}<td class=n>${dur(r.seconds)}<td class=n>${rs(r.billed)}<td class=n>${rs(r.paid)}<td class=n>${due?'<b style=color:#fcd34d>'+rs(due)+'</b>':rs(0)}<td class=n>${r.wallet==null?'-':rs(r.wallet)}<td>${r.live?`<span class="pill PENDING">parked ${dur(r.live.seconds)} &middot; ${rs(r.live.amount)}</span>`:''}<td><button onclick="showStays('${esc(r.plate)}')">Stays</button> <a href="/admin/statement/${esc(r.plate)}" target=_blank style=color:#67e8f9>Statement</a>${due&&r.plate!='UNKNOWN'?` <button class=off onclick="remind('${esc(r.plate)}')">Remind</button>`:''}</tr>`}).join('')||'<tr><td colspan=10 class=info>No plates yet.</tr>')}
async function cash(id){if(!confirm('Mark this charge as paid in cash?'))return;const j=await api('/api/admin/charge/'+id+'/cash');if(j.ok){toast('Marked paid (cash)');ledger();byPlate()}}
async function remind(p){const j=await api('/api/admin/remind/'+p);if(j.ok)toast('Reminder sent')}
async function visits(){const j=await (await fetch('/api/admin/visits')).json();$('vl').innerHTML='<tr><th>Plate<th>Owner<th>Lot &middot; slot<th>In<th>Out<th class=n>Parked<th class=n>Loop<th>How</tr>'+(j.rows.map(v=>`<tr><td>${pl(v.plate)}<td>${esc(v.owner||'-')}<td>${esc(v.lot||'')} &middot; ${esc(v.label)}<td>${when(v.entered)}<td>${v.status=='OPEN'?'<span class="pill PENDING">parked now</span>':when(v.left)}<td class=n>${v.seconds==null?'-':dur(v.seconds)}<td class=n>${v.loop_no||'-'}<td>${esc(v.source||'')}</tr>`).join('')||'<tr><td colspan=8 class=info>No entries yet.</tr>')+`<tr><td colspan=8 class=info>${j.total} visits saved in total - every entry and exit is kept for the machine-learning model, also when the video replays.</tr>`}
async function wallet(){const j=await (await fetch('/api/admin/payments/wallet')).json();$('wl').innerHTML='<tr><th>Time<th>Driver<th>What<th class=n>Amount<th class=n>Balance after</tr>'+(j.rows.map(w=>`<tr><td>${when(w.ts)}<td>${esc(w.name||'-')} ${pl(w.plate)}<td>${esc(w.kind)} <small>${esc(w.ref||'')}</small><td class=n>${w.amount<0?'-':'+'}${rs(Math.abs(w.amount))}<td class=n>${w.balance==null?'-':rs(w.balance)}</tr>`).join('')||'<tr><td colspan=5 class=info>No wallet activity yet.</tr>')}
function showStays(p){fp=p;$('fq').value='';sw('ledger');ledger()}
async function retry(id){const j=await api('/api/retry/'+id);if(j.ok)toast('Retrying payment...');ledger()}
async function check(){const j=await (await fetch('/api/admin/payments/check')).json();$('ck').innerHTML=j.lots.map(l=>`<b>${esc(l.lot)}</b> &middot; plates: ${esc(l.plate_source)} &middot; video ${dur(l.video_seconds)} &middot; ${l.charges} charges, average ${dur(l.avg_seconds)}, longest ${dur(l.longest)}<br>`+(l.issues.length?l.issues.map(i=>'&#9888; '+esc(i)).join('<br>'):'&#10003; nothing odd found')).join('<br><br>')}
async function loadTU(){const j=await (await fetch('/api/admin/test_user')).json(),u=j.user;$('tu').innerHTML=u?`Log in as <b>${esc(u.email)}</b> / <b>${esc(j.password)}</b> &middot; car ${pl(u.plate)} &middot; wallet <b>${rs(u.balance)}</b>. When this car leaves a slot its charge is taken from the wallet; if the wallet is short it stays pending and can be paid with Razorpay.`:'No test driver yet.'}
async function saveTU(){const j=await api('/api/admin/test_user',{plate:$('tp').value,balance:$('tb').value});if(j.user){toast('Test driver saved');$('tp').value=$('tb').value='';loadTU();byPlate()}}
sw('ledger');live();ledger();byPlate();loadTU();setInterval(live,2000);setInterval(ledger,5000);setInterval(byPlate,5000);wallet();setInterval(wallet,5000);visits();setInterval(visits,5000)</script>"""

PAY_ME = """<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1"><title>My payments - Smart Parking</title><style>""" + THEME + PAY_CSS + """main{max-width:860px}</style>
<script src="https://checkout.razorpay.com/v1/checkout.js"></script>
<header><a class=logo href=/nearby>&#128663; Smart<span>Parking</span></a><div class=user>__NAME__<a href=/nearby>Nearby parkings</a><a href=/bookings>My bookings</a><a href=/logout>Logout</a></div></header>
<main><h2>My payments</h2><div class=info>Parking is charged when your car leaves its slot. Wallet first; anything left over can be paid here.</div>
<div class=stats><div class=stat><small>Wallet</small><b id=bal>-</b></div><div class=stat><small>To pay</small><b id=due style=color:#fcd34d>-</b></div><div class=stat><small>My car</small><b id=pc style="font-size:15px;margin-top:8px">-</b></div></div>
<div class=card id=topup><h3>Add money to wallet</h3><div class=row><input id=ta type=number min=1 max=10000 value=500 aria-label="Amount in rupees"><button id=tbtn onclick=addMoney()>Add money</button></div><div class=info id=hint></div></div>
<div class=card><h3>Parking charges</h3><div class=sc><table id=tl></table></div></div><div class=card><h3>Wallet history</h3><div class=sc><table id=wt></table></div></div>
<div class=card id=ml><h3 style="font-size:18px;color:#e2e8f0">&#129504; Smart parking via machine learning</h3><div id=mlb class=info>Learning from your parking history...</div></div></main><div id=t></div>
<script>""" + PAY_JS + r"""
let D=null;
async function load(){D=await (await fetch('/api/my/payments')).json();$('bal').textContent=rs(D.balance);$('due').textContent=rs(D.pending);$('pc').innerHTML=pl(D.plate);
$('tbtn').textContent=D.rzp?'Add via Razorpay':'Add (test credit)';wh();$('hint').textContent=D.rzp?'':'Razorpay keys are not set, so top-ups are free test credit.';$('topup').style.display=D.wallet?'':'none';
$('tl').innerHTML='<tr><th>#<th>Time<th>Plate<th>Parked<th class=n>Amount<th>Status<th></tr>'+(D.charges.map(c=>`<tr><td>${c.id}<td>${when(c.created)}<td>${pl(c.plate)}<td>${dur(c.seconds)}<td class=n>${rs(c.amount)}<td><span class="pill ${c.status}">${c.status}</span>${c.method?' <small>'+esc(c.method)+'</small>':''}<td>${c.due?(D.wallet?`<button onclick="wal(${c.id})">Pay from wallet</button> `:'')+(D.rzp?`<button onclick="rz({charge_id:${c.id}})">Pay with Razorpay</button>`:''):''}</tr>`).join('')||'<tr><td colspan=7 class=info>No parking charges yet.</tr>');ml()}
function wh(){$('wt').innerHTML='<tr><th>Time<th>What<th class=n>Amount<th class=n>Balance</tr>'+(D.wallet_tx.map(w=>`<tr><td>${when(w.ts)}<td>${esc(w.kind)} <small>${esc(w.ref||'')}</small><td class=n>${w.amount<0?'-':'+'}${rs(Math.abs(w.amount))}<td class=n>${w.balance==null?'-':rs(w.balance)}</tr>`).join('')||'<tr><td colspan=4 class=info>No wallet activity yet.</tr>')}
let MLs='';
async function ml(){const j=await (await fetch('/api/my/plans')).json();const s=JSON.stringify(j);if(s==MLs)return;MLs=s;const u=j.usage,f=j.forecast,mx=Math.max(1,...u.daily14);
let h=`<div style="font-size:15px;color:#e2e8f0">${esc(j.message)}</div><div class=info>${esc(j.confidence)} &middot; model: ${j.model=='ridge'?'trend + weekday pattern':j.model=='flat'?'recent average':'not trained yet'} &middot; saved in the database, retrained after every stay</div>`;
h+=`<div class=stats><div class=stat><small>Stays tracked</small><b>${u.n_stays}</b></div><div class=stat><small>Total parked</small><b>${dur(u.total_secs)}</b></div><div class=stat><small>Spent so far</small><b>${rs(u.spent)}</b></div><div class=stat><small>Expected next 30 days</small><b>${dur(f.secs)}</b><small>range ${dur(f.low)} - ${dur(f.high)} &middot; about ${rs(f.cost)}</small></div></div>`;
h+=`<small>Your last 14 days</small><div class=bars>${u.daily14.map(v=>`<i style="height:${Math.round(v/mx*100)}%" title="${dur(v)}"></i>`).join('')}</div>`;
if(j.active.length)h+=j.active.map(a=>`<div class=info>&#9989; Active: <b>${esc(a.name)}</b> &middot; ${dur(a.left)} of parking left &middot; valid until ${when(a.end)}</div>`).join('');
h+=`<div class=pg>`+j.plans.map(p=>`<div class="pc${p.best?' best':''}">${p.best?'<span class="pill PAID">Best for you</span>':''}<b style="display:block;font-size:16px;margin:6px 0">${esc(p.name)}</b><div style="font-size:20px;font-weight:700">${rs(p.fee)} <small>/ ${p.days} day${p.days>1?'s':''}</small></div><small>${dur(p.secs)} of parking included &middot; ${p.off}% off &middot; extra time at the normal rate</small><div style="margin-top:8px;font-size:13px">Without plan: ${rs(p.no_plan)}<br>With plan: ${rs(p.with_plan)}<br><b style="color:${p.save>0?'#6ee7b7':'#fca5a5'}">${p.save>0?'You save about '+rs(p.save):'Not worth it yet'}</b></div><small>Pays off once you park more than ${dur(p.breakeven)}</small><div class=row style="margin-top:10px">${j.rzp?`<button onclick="rz({plan:'${p.key}'})">Buy with Razorpay</button>`:''}${j.wallet?`<button class=off onclick="buyWal('${p.key}')">Pay from wallet</button>`:''}</div></div>`).join('')+`</div>`;
if(!j.rzp)h+=`<div class=info>Razorpay keys are not set, so plans can only be paid from the wallet.</div>`;
$('mlb').innerHTML=h}
async function buyWal(k){const j=await api('/api/plan/wallet',{plan:k});if(j.ok){toast('Plan activated');MLs='';load()}}
async function wal(id){const j=await api('/api/charge/'+id+'/wallet');if(j.ok)toast('Paid from wallet');load()}
async function rz(b){const j=await api('/api/rzp/order',b);if(!j.order_id)return;
new Razorpay({key:j.key,amount:j.amount,currency:'INR',name:'Smart Parking',order_id:j.order_id,prefill:{name:j.name,email:j.email,contact:j.phone},theme:{color:'#34d399'},
handler:async r=>{const v=await api('/api/rzp/verify',r);toast(v.ok?'Payment received':'Could not verify the payment');load()}}).open()}
async function addMoney(){const a=+$('ta').value;if(!(a>=1))return toast('Enter an amount');if(D.rzp)return rz({topup:a});const j=await api('/api/topup',{amount:a});if(j.ok)toast('Wallet topped up');load()}
load();setInterval(load,5000)</script>"""

@app.route("/admin/payments")
@need_admin
def admin_payments():
    ensure_test_user()
    return PAY_ADMIN.replace("__CSRF__", session["csrf"]).replace("__NAME__", html_escape(session["name"]))

@app.route("/payments")
@need_login
def my_payments_page():
    return PAY_ME.replace("__CSRF__", session["csrf"]).replace("__NAME__", html_escape(session["name"]))


def wtx(c, uid, kind, amt, ref=""):
    """Wallet history row (written on the same connection as the balance change)."""
    b = c.execute("SELECT balance FROM users WHERE id=?", (uid,)).fetchone()
    c.execute("INSERT INTO wallet_tx(ts,user_id,kind,amount,ref,balance) VALUES(?,?,?,?,?,?)", (time.time(), uid, kind, amt, ref, b[0] if b else None))

@app.route("/api/admin/payments/wallet")
@need_admin
def pay_wallet():
    with db() as c:
        rows = [dict(r) for r in c.execute("SELECT w.ts,w.kind,w.amount,w.ref,w.balance,u.name,u.plate FROM wallet_tx w LEFT JOIN users u ON u.id=w.user_id ORDER BY w.id DESC LIMIT 300")]
    return jsonify(rows=rows)

@app.route("/api/admin/charge/<int:cid>/cash", methods=["POST"])
@need_admin
def charge_cash(cid):
    """Driver paid the guard in cash: close the charge."""
    with db() as c:
        n = c.execute("UPDATE charges SET status='PAID', run='CASH', method='CASH', paid_at=? WHERE id=? AND booking_id IS NULL AND status IN ('PENDING','FAILED')", (time.time(), cid)).rowcount
    if n: paid_alert(cid)
    return jsonify(ok=True) if n else (jsonify(error="Already paid or not payable here"), 409)

def send_sms(to10, text):
    """Twilio SMS to an Indian mobile (10 digits). Returns True when Twilio accepted it; never raises."""
    sid, tok, frm = os.getenv("TWILIO_SID"), os.getenv("TWILIO_TOKEN"), os.getenv("TWILIO_FROM")
    if not (REMIND_SMS and sid and tok and frm and norm_phone(to10)): return False
    try:
        r = requests.post(f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json", auth=(sid, tok), timeout=20, data={"To": "+91" + norm_phone(to10), "From": frm, "Body": text})
        return r.status_code in (200, 201)
    except Exception as e: print("sms failed:", e); return False

def remind_mail(c, uid, rows):
    """Remind one driver (e-mail, plus SMS when Twilio is set up). True if at least one channel was used."""
    u = c.execute("SELECT name,email,phone FROM users WHERE id=?", (uid,)).fetchone()
    if not u: return False
    tot = sum(r["amount"] for r in rows)
    sent = False
    if SMTP_HOST and u["email"]:
        notify("Parking payment pending", f"Hi {u['name']},\n\nYou have {len(rows)} unpaid parking charge(s) totalling \u20b9{tot}.\nPay from your wallet or with Razorpay: {PUBLIC_URL}/payments\n", u["email"]); sent = True
    if u["phone"] and send_sms(u["phone"], f"Smart Parking: {len(rows)} unpaid charge(s), Rs {tot}. Pay: {PUBLIC_URL}/payments"): sent = True
    return sent

@app.route("/api/admin/remind/<plate>", methods=["POST"])
@need_admin
def remind(plate):
    if not (SMTP_HOST or (os.getenv("TWILIO_SID") and os.getenv("TWILIO_TOKEN") and os.getenv("TWILIO_FROM"))): return jsonify(error="Set SMTP_HOST (e-mail) or TWILIO_SID / TWILIO_TOKEN / TWILIO_FROM (SMS) in .env to send reminders"), 400
    p = re.sub(r"[^A-Z0-9]", "", plate.upper())
    with db() as c:
        uid = plate_owner(p)
        rows = c.execute("SELECT id,amount FROM charges WHERE plate=? AND booking_id IS NULL AND status IN ('PENDING','FAILED')", (p,)).fetchall()
        if not uid: return jsonify(error="This plate has no registered driver account to e-mail"), 404
        if not rows: return jsonify(error="Nothing pending for this plate"), 409
        if not remind_mail(c, uid, rows): return jsonify(error="Could not send: the driver has no e-mail/phone that the configured channels can reach"), 502
    return jsonify(ok=True)

def remind_loop():
    """REMIND_MINS > 0: one e-mail per driver for charges that stayed pending that long."""
    while True:
        time.sleep(60)
        try:
            with db() as c:
                due = c.execute("SELECT id,user_id,amount FROM charges WHERE status='PENDING' AND booking_id IS NULL AND user_id IS NOT NULL AND reminded IS NULL AND created<?", (time.time() - REMIND_MINS * 60,)).fetchall()
                for uid in {r["user_id"] for r in due}:
                    mine = [r for r in due if r["user_id"] == uid]
                    if remind_mail(c, uid, mine): c.executemany("UPDATE charges SET reminded=? WHERE id=?", [(time.time(), r["id"]) for r in mine])
        except Exception as e: print("reminder error:", e)

@app.route("/admin/statement/<plate>")
@need_admin
def statement(plate):
    p = re.sub(r"[^A-Z0-9]", "", plate.upper()); e = html_escape
    with db() as c:
        rows = c.execute("SELECT * FROM charges WHERE COALESCE(plate,'UNKNOWN')=? ORDER BY id DESC", (p,)).fetchall()
        v = c.execute("SELECT owner,phone FROM vehicles WHERE plate=?", (p,)).fetchone()
    tot = lambda st: sum(r["amount"] for r in rows if r["status"] == st)
    body = "".join(f"<tr><td>{r['id']}<td>{time.strftime('%d %b %Y %H:%M', time.localtime(r['created']))}<td>{e(str(r['lot'] or ''))} {r['slot']}<td>{r['seconds']} s<td align=right>\u20b9{r['amount']}<td>{r['status']}<td>{e(str(r['method'] or '-'))}</tr>" for r in rows)
    return (f"<!doctype html><meta charset=utf-8><title>Statement {e(p)}</title><body style='font-family:system-ui;max-width:760px;margin:30px auto;padding:0 16px;color:#111'>"
            f"<h2>Parking statement &middot; {e(p)}</h2><p>{e(v['owner'] or '') if v else ''} {e(v['phone'] or '') if v else ''}</p>"
            f"<p>Billed <b>\u20b9{tot('PAID') + tot('PENDING') + tot('FAILED')}</b> &middot; Paid <b>\u20b9{tot('PAID')}</b> &middot; Due <b>\u20b9{tot('PENDING') + tot('FAILED')}</b></p>"
            f"<table style='width:100%;border-collapse:collapse' cellpadding=6><tr align=left><th>#<th>Date<th>Slot<th>Parked<th align=right>Amount<th>Status<th>Paid via</tr>{body}</table>"
            f"<p><button onclick=print()>Print</button></p></body>")

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
    lot = p["lot"] or FIRST
    if not p["live"] or lot not in LOTS: return render_template_string(NOLIVE, p=p)
    return render_template_string(DASH, name=session["name"], csrf=session["csrf"], admin=False, is_admin=session.get("role") == "admin", ptitle=p["name"], lot=lot, lots=LOT_KEYS)

@app.route("/admin")
@need_admin
def admin_page():
    return render_template_string(DASH, name=session["name"], csrf=session["csrf"], admin=True, is_admin=True, ptitle=f"Admin - {_cur()}", lot=_cur(), lots=LOT_KEYS)

recover_sessions()
ensure_test_user()                                  # test driver (needs an admin to exist; also created when the admin opens /admin/payments)

if os.getenv("AUTOSTART") == "1": start_workers()     # for gunicorn: AUTOSTART=1 gunicorn -w 1 --threads 8 app:app

if __name__ == "__main__":
    start_workers()
    app.run(host="0.0.0.0", port=5000, threaded=True)
