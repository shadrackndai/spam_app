import os
import uuid
from typing import Dict, List, Optional, Tuple

import pandas as pd
import psycopg2
import psycopg2.extras
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from streamlit_autorefresh import st_autorefresh

# ============================================================
# CONFIG
# ============================================================
DB_SCHEMA = "game"

HOST_PIN = os.getenv("HOST_PIN", "1234")
DEFAULT_SESSION_CODE = os.getenv("DEFAULT_SESSION_CODE", "AI2026")

LABELS = ["spam", "not_spam"]

QUIZ_MESSAGES: List[Tuple[str, str]] = [
    ("Congratulations! You have won a cash prize. Click this link to claim.", "spam"),
    ("Hey, are we still meeting today at 4?", "not_spam"),
    ("Urgent! confirm your password to avoid account suspension", "spam"),
    ("Your package arrives tomorrow. Track it in the app.", "not_spam"),
    ("Limited offer! Buy now and get 70% off", "spam"),
    ("Please share the updated budget before EOD.", "not_spam"),
]

DEFAULT_TRAINING = [
    ("Win $1,000,000 now! Click here", "spam"),
    ("Urgent! Verify your bank account immediately", "spam"),
    ("Congratulations, you have been selected for a prize", "spam"),
    ("Free airtime offer, claim now", "spam"),
    ("Limited deal! Buy now", "spam"),
    ("Your account will be suspended. Confirm password", "spam"),
    ("Hi mum, I’ll call you later", "not_spam"),
    ("Meeting tomorrow at 10am in the boardroom", "not_spam"),
    ("Please review the report and share feedback", "not_spam"),
    ("Your delivery will arrive this afternoon", "not_spam"),
    ("Let’s reschedule our appointment to Friday", "not_spam"),
]

# ============================================================
# PAGE SETUP
# ============================================================
st.set_page_config(page_title="AI in Action", page_icon="🧠", layout="wide")

st.markdown(
    """
<style>
:root{
  /* Core palette (professional blend) */
  --bg0: #0f172a;      /* slate-900 */
  --bg1: #020617;      /* near-black */
  --surface: rgba(30,41,59,0.92); /* slate-800-ish */
  --surface2: rgba(30,41,59,0.78);
  --border: rgba(148,163,184,0.22); /* slate-400-ish */
  --border2: rgba(226,232,240,0.18);

  --text: #f8fafc;     /* slate-50 */
  --muted: rgba(203,213,225,0.92); /* slate-300 */

  --blue: #3b82f6;     /* primary */
  --teal: #14b8a6;     /* secondary */
  --amber: #f59e0b;    /* accent */

  --green: #22c55e;    /* success */
  --red: #ef4444;      /* danger */
}

/* ===== Background (soft blue + teal blend, not neon) ===== */
.stApp{
  background:
    radial-gradient(900px 520px at 10% 12%, rgba(59,130,246,0.22), transparent 60%),
    radial-gradient(760px 460px at 90% 18%, rgba(20,184,166,0.18), transparent 60%),
    radial-gradient(900px 520px at 40% 92%, rgba(245,158,11,0.10), transparent 65%),
    linear-gradient(180deg, var(--bg0) 0%, var(--bg1) 100%);
}

/* ===== Remove Streamlit chrome ===== */
section[data-testid="stSidebar"] {display:none !important;}
header[data-testid="stHeader"] {display:none !important;}
footer {display:none !important;}
div[data-testid="stToolbar"] {display:none !important;}
div[data-testid="stDecoration"] {display:none !important;}
#MainMenu {visibility:hidden;}

.viewerBadge_container__1QSob,
.viewerBadge_container,
div[class^="viewerBadge_"],
div[class*="viewerBadge"] {display:none !important;}
a[href*="streamlit.io"], a[href*="streamlitapp.com"] {display:none !important;}

/* ===== Layout ===== */
.block-container{max-width:1400px; padding-top:0.75rem !important;}

/* ===== Text (force readable even on dark-mode phones) ===== */
html, body, .stApp, [class*="st-"], .stMarkdown, p, li, label {
  color: var(--text) !important;
}
small, .stCaption, .stMarkdown small {
  color: var(--muted) !important;
}

/* ===== Cards ===== */
.card, .card-tight{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 18px;
  box-shadow: 0 12px 30px rgba(0,0,0,0.35);
  backdrop-filter: blur(10px);
}
.card{padding:18px 18px; margin-bottom:12px;}
.card-tight{padding:12px 14px; margin-bottom:10px;}

/* ===== Metrics (clean dashboard look) ===== */
div[data-testid="stMetric"]{
  background: var(--surface2);
  border: 1px solid var(--border2);
  border-radius: 16px;
  padding: 10px 12px;
}
div[data-testid="stMetricValue"]{font-size: 2.1rem !important;}
div[data-testid="stMetricLabel"]{opacity: 0.9;}

/* ===== Inputs (big dark-mode fix) ===== */
input, textarea{
  background: rgba(2,6,23,0.40) !important;
  color: var(--text) !important;
  border: 1px solid rgba(148,163,184,0.28) !important;
  border-radius: 14px !important;
}
input::placeholder, textarea::placeholder{
  color: rgba(226,232,240,0.70) !important;
}

/* Selectbox shell */
div[data-testid="stSelectbox"] div[role="button"]{
  background: rgba(2,6,23,0.40) !important;
  border: 1px solid rgba(148,163,184,0.28) !important;
  border-radius: 14px !important;
  color: var(--text) !important;
}
/* Dropdown menu */
div[role="listbox"]{
  background: rgba(15,23,42,0.98) !important;
  color: var(--text) !important;
  border: 1px solid rgba(148,163,184,0.28) !important;
}

/* ===== Alerts (more visible) ===== */
div[data-testid="stAlert"]{
  border-radius: 14px !important;
  background: rgba(30,41,59,0.78) !important;
  border: 1px solid rgba(148,163,184,0.26) !important;
}
div[data-testid="stAlert"] p{color: var(--text) !important;}

/* ===== Buttons (professional, not neon) ===== */
.stButton button{
  border-radius: 14px !important;
  padding: 0.52rem 0.90rem !important;
  font-weight: 800 !important;
  line-height: 1.15 !important;
  border: 1px solid rgba(148,163,184,0.28) !important;
  transition: transform 0.06s ease, filter 0.12s ease, background 0.12s ease;
}
.stButton button:active{transform:scale(0.99);}

/* Primary button = solid blue */
button[data-testid="baseButton-primary"]{
  background: var(--blue) !important;
  color: white !important;
  border: 1px solid rgba(255,255,255,0.18) !important;
}
button[data-testid="baseButton-primary"]:hover{filter: brightness(1.10);}

/* Secondary button = slate surface */
button[data-testid="baseButton-secondary"]{
  background: rgba(30,41,59,0.85) !important;
  color: var(--text) !important;
}
button[data-testid="baseButton-secondary"]:hover{filter: brightness(1.12);}

/* ===== DataFrame ===== */
div[data-testid="stDataFrame"]{
  border-radius: 14px;
  overflow: hidden;
  border: 1px solid rgba(148,163,184,0.20);
}
div[data-testid="stDataFrame"] *{
  color: var(--text) !important;
}

/* ===== Optional utility pills ===== */
.pill-open{
  display:inline-block;
  padding: 8px 14px;
  border-radius: 999px;
  font-weight: 900;
  background: rgba(34,197,94,0.16);
  border: 1px solid rgba(34,197,94,0.55);
}
.pill-closed{
  display:inline-block;
  padding: 8px 14px;
  border-radius: 999px;
  font-weight: 900;
  background: rgba(239,68,68,0.16);
  border: 1px solid rgba(239,68,68,0.55);
}
</style>
""",
    unsafe_allow_html=True,
)



# ============================================================
# DB CONNECTION
# ============================================================
def get_db_config():
    cfg = {}
    if hasattr(st, "secrets") and "DB_HOST" in st.secrets:
        cfg["host"] = st.secrets["DB_HOST"]
        cfg["port"] = int(st.secrets.get("DB_PORT", 5432))
        cfg["dbname"] = st.secrets["DB_NAME"]
        cfg["user"] = st.secrets["DB_USER"]
        cfg["password"] = st.secrets["DB_PASSWORD"]
    else:
        cfg["host"] = os.getenv("DB_HOST", "")
        cfg["port"] = int(os.getenv("DB_PORT", "5432"))
        cfg["dbname"] = os.getenv("DB_NAME", "")
        cfg["user"] = os.getenv("DB_USER", "")
        cfg["password"] = os.getenv("DB_PASSWORD", "")
    return cfg


def db_connect():
    cfg = get_db_config()
    missing = [k for k, v in cfg.items() if not v]
    if missing:
        st.error("Database config missing. Set: DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD")
        st.stop()

    return psycopg2.connect(
        host=cfg["host"],
        port=cfg["port"],
        dbname=cfg["dbname"],
        user=cfg["user"],
        password=cfg["password"],
        connect_timeout=6,
    )


def init_db():
    sql = f"""
    CREATE SCHEMA IF NOT EXISTS {DB_SCHEMA};

    CREATE TABLE IF NOT EXISTS {DB_SCHEMA}.game_sessions (
      session_code TEXT PRIMARY KEY,
      created_at TIMESTAMPTZ DEFAULT now()
    );

    -- truth_label is used as AI label (ai_label) for this game.
    CREATE TABLE IF NOT EXISTS {DB_SCHEMA}.rounds (
      id BIGSERIAL PRIMARY KEY,
      session_code TEXT REFERENCES {DB_SCHEMA}.game_sessions(session_code) ON DELETE CASCADE,
      round_no INT NOT NULL,
      message TEXT NOT NULL,
      truth_label TEXT NOT NULL CHECK (truth_label IN ('spam','not_spam')),
      is_open BOOLEAN DEFAULT TRUE,
      started_at TIMESTAMPTZ DEFAULT now(),
      closed_at TIMESTAMPTZ
    );

    CREATE TABLE IF NOT EXISTS {DB_SCHEMA}.votes (
      id BIGSERIAL PRIMARY KEY,
      session_code TEXT NOT NULL,
      round_id BIGINT REFERENCES {DB_SCHEMA}.rounds(id) ON DELETE CASCADE,
      player_id TEXT NOT NULL,
      player_name TEXT NOT NULL,
      vote_label TEXT NOT NULL CHECK (vote_label IN ('spam','not_spam')),
      voted_at TIMESTAMPTZ DEFAULT now(),
      UNIQUE(round_id, player_id)
    );

    CREATE INDEX IF NOT EXISTS idx_votes_round ON {DB_SCHEMA}.votes(round_id);

    CREATE TABLE IF NOT EXISTS {DB_SCHEMA}.training_examples (
      id BIGSERIAL PRIMARY KEY,
      text TEXT NOT NULL,
      label TEXT NOT NULL CHECK (label IN ('spam','not_spam')),
      created_at TIMESTAMPTZ DEFAULT now()
    );

    CREATE TABLE IF NOT EXISTS {DB_SCHEMA}.training_examples_baseline (
      id BIGSERIAL PRIMARY KEY,
      text TEXT NOT NULL,
      label TEXT NOT NULL CHECK (label IN ('spam','not_spam')),
      created_at TIMESTAMPTZ DEFAULT now(),
      UNIQUE(text, label)
    );

    CREATE TABLE IF NOT EXISTS {DB_SCHEMA}.model_state (
      id INT PRIMARY KEY DEFAULT 1,
      model_version INT NOT NULL DEFAULT 0,
      updated_at TIMESTAMPTZ DEFAULT now()
    );

    INSERT INTO {DB_SCHEMA}.model_state (id, model_version)
    VALUES (1, 0)
    ON CONFLICT (id) DO NOTHING;
    """
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()

# ============================================================
# UTIL
# ============================================================
def pretty(label: str) -> str:
    return "SPAM 🚫" if label == "spam" else "NOT SPAM ✅"


def get_query_params() -> Dict[str, str]:
    try:
        qp = dict(st.query_params)
        out = {}
        for k, v in qp.items():
            if isinstance(v, list):
                out[k] = v[0] if v else ""
            else:
                out[k] = str(v)
        return out
    except Exception:
        qp = st.experimental_get_query_params()
        return {k: (v[0] if v else "") for k, v in qp.items()}


def ensure_player_id():
    if "player_id" not in st.session_state:
        st.session_state.player_id = str(uuid.uuid4())


def table_count(table: str) -> int:
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {DB_SCHEMA}.{table};")
            return int(cur.fetchone()[0])


# ============================================================
# GAME DB HELPERS
# ============================================================
def ensure_session(session_code: str):
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"INSERT INTO {DB_SCHEMA}.game_sessions(session_code) VALUES (%s) ON CONFLICT DO NOTHING",
                (session_code,),
            )
        conn.commit()


def get_current_round(session_code: str) -> Optional[dict]:
    with db_connect() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                f"""
                SELECT *
                FROM {DB_SCHEMA}.rounds
                WHERE session_code = %s
                ORDER BY id DESC
                LIMIT 1
                """,
                (session_code,),
            )
            row = cur.fetchone()
            return dict(row) if row else None


def get_next_round_no(session_code: str) -> int:
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT COALESCE(MAX(round_no), 0) + 1 FROM {DB_SCHEMA}.rounds WHERE session_code = %s",
                (session_code,),
            )
            return int(cur.fetchone()[0])


def start_round(session_code: str, message: str, ai_label: str):
    ensure_session(session_code)
    round_no = get_next_round_no(session_code)

    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                UPDATE {DB_SCHEMA}.rounds
                SET is_open = FALSE, closed_at = now()
                WHERE session_code = %s AND is_open = TRUE
                """,
                (session_code,),
            )
            cur.execute(
                f"""
                INSERT INTO {DB_SCHEMA}.rounds(session_code, round_no, message, truth_label, is_open)
                VALUES (%s, %s, %s, %s, TRUE)
                """,
                (session_code, round_no, message, ai_label),
            )
        conn.commit()


def close_voting(session_code: str):
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                UPDATE {DB_SCHEMA}.rounds
                SET is_open = FALSE, closed_at = now()
                WHERE session_code = %s AND is_open = TRUE
                """,
                (session_code,),
            )
        conn.commit()


def clear_votes(round_id: int):
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(f"DELETE FROM {DB_SCHEMA}.votes WHERE round_id = %s", (round_id,))
        conn.commit()


def reset_game_session(session_code: str):
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(f"DELETE FROM {DB_SCHEMA}.votes WHERE session_code = %s", (session_code,))
            cur.execute(f"DELETE FROM {DB_SCHEMA}.rounds WHERE session_code = %s", (session_code,))
            cur.execute(f"DELETE FROM {DB_SCHEMA}.game_sessions WHERE session_code = %s", (session_code,))
        conn.commit()


def record_vote(session_code: str, round_id: int, player_id: str, player_name: str, vote_label: str) -> Tuple[bool, str]:
    if vote_label not in LABELS:
        return False, "Invalid vote."

    name = (player_name or "").strip()
    if not name:
        return False, "Please enter your name."

    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT is_open FROM {DB_SCHEMA}.rounds WHERE id = %s AND session_code = %s",
                (round_id, session_code),
            )
            row = cur.fetchone()
            if not row:
                return False, "Round not found."
            if row[0] is False:
                return False, "Voting is closed."

            try:
                cur.execute(
                    f"""
                    INSERT INTO {DB_SCHEMA}.votes(session_code, round_id, player_id, player_name, vote_label)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (session_code, round_id, player_id, name, vote_label),
                )
            except psycopg2.errors.UniqueViolation:
                conn.rollback()
                return False, "You already voted this round."

        conn.commit()

    return True, "Vote submitted!"


def get_round_snapshot(session_code: str):
    """
    Faster: single DB connection to fetch current round + counts + votes.
    Returns: (current_round_dict_or_None, counts_dict, votes_list)
    """
    with db_connect() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                f"""
                SELECT *
                FROM {DB_SCHEMA}.rounds
                WHERE session_code = %s
                ORDER BY id DESC
                LIMIT 1
                """,
                (session_code,),
            )
            current = cur.fetchone()
            if not current:
                return None, {"spam": 0, "not_spam": 0}, []

            round_id = int(current["id"])

            # counts
            cur.execute(
                f"""
                SELECT vote_label, COUNT(*)::int
                FROM {DB_SCHEMA}.votes
                WHERE round_id = %s
                GROUP BY vote_label
                """,
                (round_id,),
            )
            rows = cur.fetchall()
            counts = {"spam": 0, "not_spam": 0}
            for r in rows:
                counts[r["vote_label"]] = r["count"]

            # votes list
            cur.execute(
                f"""
                SELECT player_name, vote_label, voted_at
                FROM {DB_SCHEMA}.votes
                WHERE round_id = %s
                ORDER BY voted_at ASC
                """,
                (round_id,),
            )
            votes = cur.fetchall() or []

            return dict(current), counts, [dict(v) for v in votes]


def player_already_voted(round_id: int, player_id: str) -> Optional[str]:
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT vote_label FROM {DB_SCHEMA}.votes WHERE round_id = %s AND player_id = %s",
                (round_id, player_id),
            )
            row = cur.fetchone()
            return row[0] if row else None


def majority_from_counts(counts: Dict[str, int]) -> Optional[str]:
    if counts["spam"] == 0 and counts["not_spam"] == 0:
        return None
    if counts["spam"] > counts["not_spam"]:
        return "spam"
    if counts["not_spam"] > counts["spam"]:
        return "not_spam"
    return None


# ============================================================
# ML HELPERS
# ============================================================
def seed_baseline_if_empty():
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {DB_SCHEMA}.training_examples_baseline;")
            base_n = int(cur.fetchone()[0])

            if base_n == 0:
                cur.execute(f"SELECT COUNT(*) FROM {DB_SCHEMA}.training_examples;")
                train_n = int(cur.fetchone()[0])

                if train_n > 0:
                    cur.execute(
                        f"""
                        INSERT INTO {DB_SCHEMA}.training_examples_baseline(text, label)
                        SELECT text, label FROM {DB_SCHEMA}.training_examples
                        ON CONFLICT (text, label) DO NOTHING
                        """
                    )
                else:
                    cur.executemany(
                        f"INSERT INTO {DB_SCHEMA}.training_examples_baseline(text, label) VALUES (%s, %s)",
                        DEFAULT_TRAINING,
                    )
        conn.commit()


def seed_training_if_empty():
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {DB_SCHEMA}.training_examples;")
            n = int(cur.fetchone()[0])
            if n == 0:
                cur.execute(
                    f"""
                    INSERT INTO {DB_SCHEMA}.training_examples(text, label)
                    SELECT text, label FROM {DB_SCHEMA}.training_examples_baseline
                    ORDER BY id ASC
                    """
                )
        conn.commit()


def load_training_df() -> pd.DataFrame:
    with db_connect() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(f"SELECT text, label FROM {DB_SCHEMA}.training_examples ORDER BY id ASC;")
            rows = cur.fetchall()
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["text", "label"])


def build_model_from_df(df: pd.DataFrame) -> Pipeline:
    model = Pipeline(
        [
            ("vectorizer", CountVectorizer(lowercase=True, stop_words="english")),
            ("clf", MultinomialNB()),
        ]
    )
    model.fit(df["text"], df["label"])
    return model


def get_model_version() -> int:
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(f"SELECT model_version FROM {DB_SCHEMA}.model_state WHERE id=1;")
            return int(cur.fetchone()[0])


def bump_model_version():
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                UPDATE {DB_SCHEMA}.model_state
                SET model_version = model_version + 1, updated_at = now()
                WHERE id = 1;
                """
            )
        conn.commit()


@st.cache_resource
def get_cached_model(version: int) -> Pipeline:
    # Only rebuild when model_version changes (i.e., when Retrain is clicked)
    df = load_training_df()
    if df.empty:
        df = pd.DataFrame(DEFAULT_TRAINING, columns=["text", "label"])
    return build_model_from_df(df)


def explain_prediction(model: Pipeline, text: str, top_n: int = 6):
    vec: CountVectorizer = model.named_steps["vectorizer"]
    clf: MultinomialNB = model.named_steps["clf"]

    X = vec.transform([text])
    pred = clf.predict(X)[0]
    proba = clf.predict_proba(X)[0]

    classes = list(clf.classes_)
    pred_idx = classes.index(pred)
    conf = float(proba[pred_idx])

    feature_names = vec.get_feature_names_out()
    nz = X.nonzero()[1]
    scored = [(feature_names[j], float(clf.feature_log_prob_[pred_idx, j])) for j in nz]
    scored.sort(key=lambda x: x[1], reverse=True)
    top_words = [w for w, _ in scored[:top_n]]
    return pred, conf, top_words


def poison_flip_labels(k: int = 10):
    # IMPORTANT: this now ONLY changes data. It does NOT retrain automatically.
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT id, label FROM {DB_SCHEMA}.training_examples ORDER BY random() LIMIT %s;",
                (k,),
            )
            rows = cur.fetchall()
            for _id, label in rows:
                new_label = "spam" if label == "not_spam" else "not_spam"
                cur.execute(
                    f"UPDATE {DB_SCHEMA}.training_examples SET label=%s WHERE id=%s;",
                    (new_label, _id),
                )
        conn.commit()


def poison_inject_wrong():
    poison = [
        ("Urgent: meeting moved to 3pm", "spam"),
        ("Free this afternoon for a quick call?", "spam"),
        ("Click the report link in Teams", "spam"),
        ("Congratulations on your promotion", "spam"),
        ("We would like to inform you that the Mental Health Day", "spam"),
        ("Win the contract by submitting the proposal", "spam"),
    ]
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.executemany(
                f"INSERT INTO {DB_SCHEMA}.training_examples(text, label) VALUES (%s, %s)",
                poison,
            )
        conn.commit()


def reset_training_from_baseline():
    # IMPORTANT: this resets data but does NOT retrain automatically.
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(f"TRUNCATE TABLE {DB_SCHEMA}.training_examples RESTART IDENTITY;")
            cur.execute(
                f"""
                INSERT INTO {DB_SCHEMA}.training_examples(text, label)
                SELECT text, label FROM {DB_SCHEMA}.training_examples_baseline
                ORDER BY id ASC
                """
            )
        conn.commit()


# ============================================================
# BOOTSTRAP (run once per browser session to speed up clicks)
# ============================================================
ensure_player_id()
params = get_query_params()
role = params.get("role", "").lower()
pin = params.get("pin", "")
session_code = (params.get("session", "") or DEFAULT_SESSION_CODE).strip()

if "bootstrapped" not in st.session_state:
    init_db()
    seed_baseline_if_empty()
    seed_training_if_empty()
    st.session_state.bootstrapped = True

# ============================================================
# AUTO-REFRESH helper (robust start/stop)
# ============================================================
def auto_refresh_control(label: str, default_on: bool, interval_ms: int, key_prefix: str):
    """
    Renders a toggle and performs autorefresh when enabled.
    Uses version bump on OFF to fully stop scheduled refreshes.
    """
    if f"{key_prefix}_ver" not in st.session_state:
        st.session_state[f"{key_prefix}_ver"] = 0
    if f"{key_prefix}_prev" not in st.session_state:
        st.session_state[f"{key_prefix}_prev"] = False

    enabled = st.toggle(label, value=default_on, key=f"{key_prefix}_toggle")

    # Detect ON->OFF and bump version to kill the old component
    if st.session_state[f"{key_prefix}_prev"] and not enabled:
        st.session_state[f"{key_prefix}_ver"] += 1
    st.session_state[f"{key_prefix}_prev"] = enabled

    if enabled:
        st_autorefresh(interval=interval_ms, key=f"{key_prefix}_ar_{st.session_state[f'{key_prefix}_ver']}")
    return enabled

# ============================================================
# HOST VIEW
# ============================================================
if role == "host":
    if pin != HOST_PIN:
        st.error("Host access denied. Use: ?role=host&pin=YOURPIN&session=SESSIONCODE")
        st.stop()

    # =========================
    # HOST-ONLY DASHBOARD POLISH (brighter + projector friendly)
    # =========================
    
    st.markdown("""
    <style>
    /* Host: projector layout only (do NOT override global theme colors) */
    .block-container {max-width: 1600px !important; padding-top: 0.6rem !important;}
    h1 {font-size: 2.8rem !important; letter-spacing: -0.02em;}
    h2 {font-size: 2.0rem !important;}
    h3 {font-size: 1.45rem !important;}

    /* Scoreboard sizing only */
    div[data-testid="stMetricValue"]{font-size: 2.5rem !important;}
    div[data-testid="stMetricLabel"]{font-size: 1.05rem !important; opacity: 0.9;}

    /* Button sizing only */
    .stButton button{
    border-radius: 16px !important;
    padding: 0.54rem 0.95rem !important;
    }
    </style>
    """, unsafe_allow_html=True)


    # =========================
    # HERO HEADER (more “live show”)
    # =========================
    st.markdown(
        f"""
<div class="card">
  <div style="display:flex; align-items:center; justify-content:space-between; gap:12px; flex-wrap:wrap;">
    <div>
      <div style="font-size:44px; font-weight:900; line-height:1.05;">
        AI in Action
      </div>
      <div style="margin-top:6px; font-size:16px; opacity:0.92;">
        🧪 Model Lab + 🎮 Beat the AI — Live Demo Dashboard
      </div>
    </div>
    <div style="
      padding:10px 14px; border-radius:999px;
      background:rgba(255,255,255,0.18);
      border:1px solid rgba(255,255,255,0.26);
      font-weight:900;
      ">
      Session: <span style="color:#a855f7;">{session_code}</span>
    </div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    tab_lab, tab_game = st.tabs(["🧪 Model Lab", "🎮 Beat the AI"])

    # ----------------------------
    # TAB 1: MODEL LAB
    # ----------------------------
    with tab_lab:
        st.markdown('<div class="card-tight">', unsafe_allow_html=True)
        st.markdown("### 🧪 Model Lab")
        st.caption("Change data, then click Retrain to rebuild the model (only retrains when you click Retrain).")
        st.markdown("</div>", unsafe_allow_html=True)

        # Auto-refresh in lab (slower)
        auto_refresh_control("🔄 Auto-refresh (Lab)", default_on=True, interval_ms=5000, key_prefix="host_lab")

        train_n = table_count("training_examples")
        base_n = table_count("training_examples_baseline")
        ver = get_model_version()

        a, b, c = st.columns(3)
        a.metric("Training rows", train_n)
        b.metric("Baseline rows", base_n)
        c.metric("Model version", ver)

        st.write("")
        st.markdown("### 🧪 Try a message")

        lab_use_custom = st.toggle("✍️ Type a custom message", value=True, key="lab_use_custom")

        if lab_use_custom:
            test_text = st.text_area(
                "Message to test",
                placeholder="Type any message here…",
                height=90,
                key="lab_test_text",
            ).strip()
        else:
            lab_pick = st.selectbox(
                "Pick a preset message",
                list(range(len(QUIZ_MESSAGES))),
                format_func=lambda i: QUIZ_MESSAGES[i][0],
                key="lab_preset_select",
            )
            test_text = QUIZ_MESSAGES[lab_pick][0]

        model = get_cached_model(ver)

        if test_text:
            pred, conf, words = explain_prediction(model, test_text)

            # Brighter, color-coded prediction card
            if pred == "spam":
                st.markdown(
                    f"""
                    <div class="card-tight" style="border-color: rgba(239,68,68,0.70) !important; background: rgba(239,68,68,0.20) !important;">
                      <div style="font-size:16px; opacity:0.9;">🤖 Model Prediction</div>
                      <div style="font-size:30px; font-weight:900; color:#ef4444;">🚫 SPAM</div>
                      <div style="opacity:0.9;">Confidence: <b>{conf:.2f}</b></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="card-tight" style="border-color: rgba(34,197,94,0.70) !important; background: rgba(34,197,94,0.20) !important;">
                      <div style="font-size:16px; opacity:0.9;">🤖 Model Prediction</div>
                      <div style="font-size:30px; font-weight:900; color:#22c55e;">✅ NOT SPAM</div>
                      <div style="opacity:0.9;">Confidence: <b>{conf:.2f}</b></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            if words:
                st.write("Top keywords:", ", ".join([f"`{w}`" for w in words]))
        else:
            st.info("Type a message above to see the model’s prediction, confidence, and keywords.")

        st.write("")
        st.markdown('<div class="card-tight">', unsafe_allow_html=True)
        st.markdown("### 🧠 Controls")

        b1, b2, b3, b4 = st.columns(4)

        with b1:
            if st.button("🔁 Retrain", type="primary", key="ml_retrain"):
                with st.spinner("Retraining model..."):
                    bump_model_version()
                st.success("Model retrained.")
                st.rerun()

        with b2:
            if st.button("💥 Flip labels", key="ml_flip"):
                with st.spinner("Poisoning data..."):
                    poison_flip_labels(k=10)
                st.warning("Labels flipped. Now click Retrain to apply changes to the model.")

        with b3:
            if st.button("☠️ Inject bad data", key="ml_poison"):
                with st.spinner("Injecting bad data..."):
                    poison_inject_wrong()
                st.warning("Bad rows injected. Now click Retrain to apply changes to the model.")

        with b4:
            if st.button("↩️ Reset data", key="ml_reset"):
                with st.spinner("Restoring baseline data..."):
                    reset_training_from_baseline()
                st.success("Training data restored. Now click Retrain to rebuild the model.")

        st.markdown("</div>", unsafe_allow_html=True)

    # ----------------------------
    # TAB 2: BEAT THE AI (GAME)
    # ----------------------------
    with tab_game:
        # Control bar
        st.markdown('<div class="card-tight">', unsafe_allow_html=True)
        st.markdown("### 🎮 Beat the AI")
        st.caption("Auto-refresh is ON by default. Close voting to reveal the AI prediction.")

        # Auto-refresh in game
        auto_refresh_control("🔄 Auto-refresh (Game)", default_on=True, interval_ms=2000, key_prefix="host_game")

        g1, g2, g3 = st.columns(3)
        with g1:
            if st.button("🧹 Clear votes", use_container_width=True, key="game_clear_votes"):
                cur_round = get_current_round(session_code)
                if cur_round:
                    with st.spinner("Clearing votes..."):
                        clear_votes(int(cur_round["id"]))
                    st.warning("Votes cleared.")
                    st.rerun()
                else:
                    st.info("No round yet.")

        with g2:
            if st.button("🔁 Restart game", use_container_width=True, key="game_restart"):
                with st.spinner("Restarting..."):
                    reset_game_session(session_code)
                st.success("Game reset.")
                st.rerun()

        with g3:
            if st.button("🔄 Refresh now", use_container_width=True, key="game_refresh_now"):
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)
        st.write("")

        # Message input
        use_custom = st.toggle("✍️ Use custom message", value=True, key="game_custom_toggle")

        if use_custom:
            msg = st.text_area(
                "Message for this round",
                placeholder="Type any message here…",
                height=90,
                key="game_msg_text",
            ).strip()
        else:
            pick = st.selectbox(
                "Pick a preset message",
                list(range(len(QUIZ_MESSAGES))),
                format_func=lambda i: QUIZ_MESSAGES[i][0],
                key="game_preset_select",
            )
            msg = QUIZ_MESSAGES[pick][0]

        ver = get_model_version()
        model = get_cached_model(ver)

        c1, c2 = st.columns(2)
        with c1:
            if st.button("▶️ Start round", type="primary", use_container_width=True, key="game_start_round"):
                if not msg:
                    st.error("Please type a message (or choose a preset) first.")
                else:
                    with st.spinner("Starting round..."):
                        ai_pred, ai_conf, _ = explain_prediction(model, msg)
                        start_round(session_code, msg, ai_pred)
                    st.success("Round started. AI prediction is hidden until voting closes.")
                    st.caption(f"(Host note: confidence was {ai_conf:.2f})")
                    st.rerun()

        with c2:
            if st.button("⏹️ Close voting", use_container_width=True, key="game_close_voting"):
                with st.spinner("Closing voting..."):
                    close_voting(session_code)
                st.warning("Voting closed.")
                st.rerun()

        st.write("")
        st.markdown('<div class="card">', unsafe_allow_html=True)

        current, counts, votes = get_round_snapshot(session_code)
        if not current:
            st.info("No round yet. Start one above.")
            st.markdown("</div>", unsafe_allow_html=True)
            st.stop()

        total = counts["spam"] + counts["not_spam"]
        ai_label = current["truth_label"]

        st.markdown(f"## 🔔 Round #{current['round_no']}")
        st.markdown(f"**Message:** {current['message']}")

        # Bright voting status pill
        if current["is_open"]:
            st.markdown(
                "<div style='display:inline-block;background:rgba(34,197,94,0.25);border:1px solid rgba(34,197,94,0.60);padding:8px 14px;border-radius:999px;font-weight:900;'>🟢 VOTING OPEN</div>",
                unsafe_allow_html=True,
            )
            st.write("")
            st.info("AI prediction is hidden until voting closes.")
        else:
            st.markdown(
                "<div style='display:inline-block;background:rgba(239,68,68,0.25);border:1px solid rgba(239,68,68,0.60);padding:8px 14px;border-radius:999px;font-weight:900;'>🔴 VOTING CLOSED</div>",
                unsafe_allow_html=True,
            )
            st.write("")
            # Bright AI prediction reveal card
            if ai_label == "spam":
                st.markdown(
                    "<div class='card-tight' style='border-color: rgba(239,68,68,0.70) !important; background: rgba(239,68,68,0.20) !important;'><div style='font-size:16px; opacity:0.9;'>🤖 AI Prediction</div><div style='font-size:30px; font-weight:900; color:#ef4444;'>🚫 SPAM</div></div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    "<div class='card-tight' style='border-color: rgba(34,197,94,0.70) !important; background: rgba(34,197,94,0.20) !important;'><div style='font-size:16px; opacity:0.9;'>🤖 AI Prediction</div><div style='font-size:30px; font-weight:900; color:#22c55e;'>✅ NOT SPAM</div></div>",
                    unsafe_allow_html=True,
                )

        # Scoreboard metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Total votes", total)
        m2.metric("SPAM 🚫", counts["spam"])
        m3.metric("NOT SPAM ✅", counts["not_spam"])

        if (not current["is_open"]) and total > 0:
            maj = majority_from_counts(counts)
            agree = counts.get(ai_label, 0)
            agree_pct = agree / total if total else 0

            st.write("")
            st.markdown("### 🧾 Humans vs AI")
            if maj is None:
                st.warning("Human votes are tied — no majority.")
            else:
                st.write(f"**Human majority:** {pretty(maj)}")
            st.write(f"**AI prediction:** {pretty(ai_label)}")
            st.write(f"**Agreement with AI:** {agree}/{total} (**{agree_pct:.0%}**)")

        st.write("")
        st.markdown("### 🗳️ Individual responses")
        if not votes:
            st.info("No votes yet.")
        else:
            rows = []
            for v in votes:
                rows.append(
                    {
                        "Name": v["player_name"],
                        "Vote": pretty(v["vote_label"]),
                      #  "Time": v["voted_at"].strftime("%H:%M:%S") if v.get("voted_at") else "",
                    }
                )
            st.dataframe(rows, use_container_width=True, hide_index=True)

        st.markdown("</div>", unsafe_allow_html=True)

    st.stop()


# ============================================================
# PLAYER VIEW
# ============================================================
st.markdown(
    """
<div class="card">
  <div style="font-size: 28px; font-weight: 900; line-height: 1.1;">
    HUMANS vs AI
  </div>
  <div style="margin-top: 6px; opacity: 0.85;">
    Auto-refresh is ON. Vote once per round. After voting closes, you’ll see the AI prediction.
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# Player auto-refresh (slower to reduce load)
auto_refresh_control("🔄 Auto-refresh", default_on=True, interval_ms=2500, key_prefix="player")

name = st.text_input("Your name:", placeholder="e.g. John", max_chars=40, key="player_name")

current = get_current_round(session_code)
if not current:
    st.info("Waiting for the host to start a round…")
    st.stop()

round_id = int(current["id"])
st.markdown(f"### Round {current['round_no']}")
st.write(f"**Message:** {current['message']}")
st.write(f"**Voting:** {'🟢 OPEN' if current['is_open'] else '🔴 CLOSED'}")

ai_label = current["truth_label"]
if current["is_open"]:
    st.info("AI prediction is hidden until voting closes.")
else:
    st.success(f"🤖 AI prediction: **{pretty(ai_label)}**")

# If voting is closed, we DO NOT show the "already voted" message.
if not current["is_open"]:
    st.caption("Voting is closed. Wait for the next round.")
    st.stop()

# Voting is open: prevent double voting
existing = player_already_voted(round_id, st.session_state.player_id)
if existing:
    st.info("✅ Vote received. Please wait for voting to close.")
    st.stop()

st.markdown("### Choose your answer:")
colA, colB = st.columns(2)

with colA:
    if st.button("🚫 SPAM", type="primary", use_container_width=True, key="player_vote_spam"):
        ok, msg = record_vote(session_code, round_id, st.session_state.player_id, name, "spam")
        st.success(msg) if ok else st.error(msg)
        st.rerun()

with colB:
    if st.button("✅ NOT SPAM", type="primary", use_container_width=True, key="player_vote_not_spam"):
        ok, msg = record_vote(session_code, round_id, st.session_state.player_id, name, "not_spam")
        st.success(msg) if ok else st.error(msg)
        st.rerun()

st.caption("Tip: If something looks stuck, toggle auto-refresh off/on once.")
