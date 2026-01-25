import os
import time
import uuid
from typing import Dict, List, Optional, Tuple

import psycopg2
import psycopg2.extras
import streamlit as st
from streamlit_autorefresh import st_autorefresh


# =========================
# CONFIG
# =========================
DB_SCHEMA = "game"

HOST_PIN = os.getenv("HOST_PIN", "1234")  # set in secrets/env for safety
DEFAULT_SESSION_CODE = os.getenv("DEFAULT_SESSION_CODE", "AI2026")

LABELS = ["spam", "not_spam"]

QUIZ_MESSAGES: List[Tuple[str, str]] = [
    ("Congratulations! You have won a cash prize. Click this link to claim.", "spam"),
    ("Hey, are we still meeting today at 4?", "not_spam"),
    ("Urgent: confirm your password to avoid account suspension", "spam"),
    ("Your package arrives tomorrow. Track it in the app.", "not_spam"),
    ("Limited offer! Buy now and get 70% off", "spam"),
    ("Please share the updated budget before EOD.", "not_spam"),
]

# =========================
# DB CONNECTION
# =========================
def get_db_config():
    # Prefer Streamlit secrets if available, else environment variables
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
        st.error(
            "Database config missing. Set these secrets/env vars: "
            + ", ".join(["DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASSWORD"])
        )
        st.stop()

    conn = psycopg2.connect(
        host=cfg["host"],
        port=cfg["port"],
        dbname=cfg["dbname"],
        user=cfg["user"],
        password=cfg["password"],
        connect_timeout=10,
    )

    # ✅ Force all unqualified tables to use schema "game"
  #  with conn.cursor() as cur:
   #     cur.execute("SET search_path TO game;")
  #  conn.commit()

    return conn


def init_db():
    sql = """
    CREATE TABLE IF NOT EXISTS game.game_sessions (
      session_code TEXT PRIMARY KEY,
      created_at TIMESTAMPTZ DEFAULT now()
    );

    CREATE TABLE IF NOT EXISTS game.rounds (
      id BIGSERIAL PRIMARY KEY,
      session_code TEXT REFERENCES game.game_sessions(session_code) ON DELETE CASCADE,
      round_no INT NOT NULL,
      message TEXT NOT NULL,
      truth_label TEXT NOT NULL,
      is_open BOOLEAN DEFAULT TRUE,
      started_at TIMESTAMPTZ DEFAULT now(),
      closed_at TIMESTAMPTZ
    );

    CREATE TABLE IF NOT EXISTS game.votes (
      id BIGSERIAL PRIMARY KEY,
      session_code TEXT NOT NULL,
      round_id BIGINT REFERENCES game.rounds(id) ON DELETE CASCADE,
      player_id TEXT NOT NULL,
      player_name TEXT,
      vote_label TEXT NOT NULL,
      voted_at TIMESTAMPTZ DEFAULT now(),
      UNIQUE(round_id, player_id)
    );

    CREATE INDEX IF NOT EXISTS idx_votes_round ON game.votes(round_id);

    """
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()


# =========================
# UTIL
# =========================
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


def start_round(session_code: str, message: str, truth: str):
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
                RETURNING id
                """,
                (session_code, round_no, message, truth),
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


def record_vote(session_code: str, round_id: int, player_id: str, player_name: str, vote_label: str) -> Tuple[bool, str]:
    if vote_label not in LABELS:
        return False, "Invalid vote."

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
                    (session_code, round_id, player_id, player_name.strip() or "Anonymous", vote_label),
                )
            except psycopg2.errors.UniqueViolation:
                conn.rollback()
                return False, "You already voted this round."

        conn.commit()
    return True, "Vote submitted!"


def get_votes(round_id: int) -> List[dict]:
    with db_connect() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                f"""
                SELECT player_name, vote_label, voted_at
                FROM {DB_SCHEMA}.votes
                WHERE round_id = %s
                ORDER BY voted_at ASC
                """,
                (round_id,),
            )
            rows = cur.fetchall()
            return [dict(r) for r in rows]


def vote_counts(round_id: int) -> Dict[str, int]:
    with db_connect() as conn:
        with conn.cursor() as cur:
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
    for label, c in rows:
        counts[label] = c
    return counts


def player_already_voted(round_id: int, player_id: str) -> Optional[str]:
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT vote_label FROM {DB_SCHEMA}.votes WHERE round_id = %s AND player_id = %s",
                (round_id, player_id),
            )
            row = cur.fetchone()
            return row[0] if row else None

def reset_game_session(session_code: str):
    with db_connect() as conn:
        with conn.cursor() as cur:
            # Delete votes first (FK safety)
            cur.execute(f"DELETE FROM game.votes WHERE session_code = %s", (session_code,))
            cur.execute(f"DELETE FROM game.rounds WHERE session_code = %s", (session_code,))
            cur.execute(f"DELETE FROM game.game_sessions WHERE session_code = %s", (session_code,))
        conn.commit()

# =========================
# UI
# =========================
st.set_page_config(page_title="Humans vs AI", page_icon="📱", layout="wide")
ensure_player_id()
st.markdown("""
<style>
/* Hide Streamlit sidebar */
section[data-testid="stSidebar"] {display: none !important;}

/* Hide top bar / menu */
header[data-testid="stHeader"] {display: none !important;}

/* Hide footer + deployed branding */
footer {display: none !important;}
div[data-testid="stToolbar"] {display: none !important;}
div[data-testid="stDecoration"] {display: none !important;}

/* Hide the “Made with Streamlit / hosted by” style elements (Cloud) */
a[href*="streamlit.io"], a[href*="streamlitapp.com"] {display: none !important;}
.viewerBadge_container__1QSob {display: none !important;}  /* Streamlit Cloud badge class (common) */
#MainMenu {visibility: hidden;}  /* fallback */
</style>
""", unsafe_allow_html=True)



# Init tables (safe to run; it’s idempotent)
init_db()

params = get_query_params()
role = params.get("role", "").lower()
pin = params.get("pin", "")
session_code = (params.get("session", "") or DEFAULT_SESSION_CODE).strip()


# st.caption("Everyone votes individually from their phone. Host controls the rounds and sees all responses live.")

# =========================
# HOST VIEW
# =========================
if role == "host":
    if pin != HOST_PIN:
        st.error("Host access denied. Use: ?role=host&pin=YOURPIN&session=SESSIONCODE")
        st.stop()

    # =========================
    # Projector / Dashboard CSS (Host Only)
    # =========================
    st.markdown("""
    <style>
    /* Hide Streamlit sidebar */
    section[data-testid="stSidebar"] {display: none !important;}

    /* Hide top menu + toolbar */
    header[data-testid="stHeader"] {display: none !important;}

    /* Hide footer */
    footer {display: none !important;}

    /* Wider page + reduce padding now that header is gone */
    .block-container {padding-top: 1rem !important; padding-bottom: 2rem; max-width: 1400px;}

    /* Big typography */
    h1 {font-size: 2.4rem !important;}
    h2 {font-size: 1.8rem !important;}
    h3 {font-size: 1.4rem !important;}
    p, li, label, .stMarkdown {font-size: 1.15rem !important;}

    /* Bigger metrics */
    div[data-testid="stMetricValue"] {font-size: 2.1rem !important;}
    div[data-testid="stMetricLabel"] {font-size: 1.0rem !important; opacity: 0.85;}

    /* Bigger buttons */
    .stButton button, .stDownloadButton button {
      font-size: 1.05rem !important;
      padding: 0.8rem 1.0rem !important;
      border-radius: 14px !important;
    }

    /* Expanders */
    div[data-testid="stExpander"] details summary {font-size: 1.15rem !important;}

    /* Table text bigger */
    div[data-testid="stDataFrame"] * {font-size: 1.05rem !important;}
    </style>
    """, unsafe_allow_html=True)

    # =========================
    # Header
    # =========================
    st.markdown("## 🧑‍🏫 Host Dashboard (Projector View)")
    st.caption("Controls on the left, live results on the right.")

    # =========================
    # Top row: Auto-refresh + Restart (same row)
    # =========================
    top1, top2, top3 = st.columns([1.2, 1.2, 2.6])

    with top1:
        auto = st.toggle("🔄 Auto-refresh", value=True)
        if auto:
            st_autorefresh(interval=5000, key="host_autorefresh")  # 5s; change to 2000 if you want faster

    with top2:
        if st.button("🔁 Restart game", type="primary", use_container_width=True):
            reset_game_session(session_code)
            st.success("Game reset. Ready to start fresh.")
            st.rerun()

    with top3:
        st.markdown(f"### Session: `{session_code}`")
        st.caption("Share the Player link with QR code. Keep the Host link private.")

    st.divider()

    # =========================
    # Two-column dashboard
    # =========================
    left, right = st.columns([1.05, 1.6], gap="large")

    # ---------- LEFT: Controls + Links ----------
    with left:
        st.markdown("### 🎛️ Round Controls")

        pick = st.selectbox(
            "Pick a message",
            list(range(len(QUIZ_MESSAGES))),
            format_func=lambda i: QUIZ_MESSAGES[i][0]
        )
        msg, truth = QUIZ_MESSAGES[pick]

        b1, b2 = st.columns(2)
        with b1:
            if st.button("▶️ Start round", use_container_width=True):
                start_round(session_code, msg, truth)
                st.success("Round started.")
                st.rerun()
        with b2:
            if st.button("⏹️ Close voting", use_container_width=True):
                close_voting(session_code)
                st.warning("Voting closed.")
                st.rerun()

        st.divider()
        st.markdown("### 🔗 Links")
        st.code(f"PLAYER: https://aicareerfairspamapp.streamlit.app/?session={session_code}", language="text")
        st.code(f"HOST:   https://aicareerfairspamapp.streamlit.app/?role=host&pin={HOST_PIN}&session={session_code}", language="text")

        st.info("Tip: For a clean full-screen projector view, press **F11** in your browser.")

    # ---------- RIGHT: Live Round + Results ----------
    with right:
        current = get_current_round(session_code)

        if not current:
            st.info("No round yet. Start one from the left panel.")
            st.stop()

        round_id = int(current["id"])
        counts = vote_counts(round_id)
        total = counts["spam"] + counts["not_spam"]

        st.markdown(f"## 🔔 LIVE ROUND #{current['round_no']}")
        st.markdown(f"**Message:** {current['message']}")
        st.markdown(f"**Voting:** {'🟢 OPEN' if current['is_open'] else '🔴 CLOSED'}")

        # Hide ground truth until voting closes
        if current["is_open"]:
            st.info("Answer hidden — close voting to reveal.")
        else:
            st.success(f"✅ Correct answer: **{pretty(current['truth_label'])}**")

        m1, m2, m3 = st.columns(3)
        m1.metric("Total votes", total)
        m2.metric("SPAM 🚫", counts["spam"])
        m3.metric("NOT SPAM ✅", counts["not_spam"])

        st.divider()
        st.markdown("### 🗳️ Individual responses")
        votes = get_votes(round_id)
        if not votes:
            st.info("No votes yet.")
        else:
            rows = []
            for v in votes:
                rows.append({
                    "Name": v["player_name"],
                    "Vote": pretty(v["vote_label"]),
                    "Time": v["voted_at"].strftime("%H:%M:%S") if v.get("voted_at") else ""
                })
            st.dataframe(rows, use_container_width=True, hide_index=True)

        # Bottom row actions
        action1, action2 = st.columns(2)
        with action1:
            if st.button("🧹 Clear votes (this round)", use_container_width=True):
                clear_votes(round_id)
                st.warning("Votes cleared.")
                st.rerun()
        with action2:
            st.caption("Players can refresh if they don’t see the new round.")

    st.stop()


# =========================
# PLAYER VIEW
# =========================
st.title("HUMANS vs AI ")
st.subheader("Player View")

st.write(f"Session: `{session_code}`")
name = st.text_input("Your name:", placeholder="e.g. John")

current = get_current_round(session_code)
if not current:
    st.info("Waiting for the host to start a round…")
    st.stop()

round_id = int(current["id"])
st.markdown(f"### Round {current['round_no']}")
st.write(f"**Message:** {current['message']}")
st.write(f"**Voting:** {'🟢 OPEN' if current['is_open'] else '🔴 CLOSED'}")

existing = player_already_voted(round_id, st.session_state.player_id)
if existing:
    st.success(f"You already voted: **{pretty(existing)}**")
    if st.button("🔄 Refresh"):
        st.rerun()
    st.stop()

if not current["is_open"]:
    st.warning("Voting is closed. Wait for the next round.")
    if st.button("🔄 Refresh"):
        st.rerun()
    st.stop()

st.write("Choose your answer:")

colA, colB = st.columns(2)
with colA:
    if st.button("🚫 SPAM", use_container_width=True):
        ok, msg = record_vote(session_code, round_id, st.session_state.player_id, name, "spam")
        st.success(msg) if ok else st.error(msg)
        st.rerun()

with colB:
    if st.button("✅ NOT SPAM", use_container_width=True):
        ok, msg = record_vote(session_code, round_id, st.session_state.player_id, name, "not_spam")
        st.success(msg) if ok else st.error(msg)
        st.rerun()

st.caption("If the host starts a new round, tap Refresh to load the next message.")
