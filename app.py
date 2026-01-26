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

# =========================
# CONFIG
# =========================
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

# Used only if training_examples table is empty
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

# =========================
# PAGE SETUP
# =========================
st.set_page_config(page_title="Humans vs AI", page_icon="📱", layout="wide")

# ---- Global "clean chrome" CSS (host + player) ----
st.markdown(
    """
<style>
/* Hide sidebar + header + footer */
section[data-testid="stSidebar"] {display: none !important;}
header[data-testid="stHeader"] {display: none !important;}
footer {display: none !important;}
div[data-testid="stToolbar"] {display: none !important;}
div[data-testid="stDecoration"] {display: none !important;}
#MainMenu {visibility: hidden;}

/* Streamlit Cloud floating badges (bottom-right) */
.viewerBadge_container__1QSob {display: none !important;}
.viewerBadge_container {display: none !important;}
div[class^="viewerBadge_"] {display: none !important;}
div[class*="viewerBadge"] {display: none !important;}

a[href*="streamlit.io"] {display: none !important;}
a[href*="streamlitapp.com"] {display: none !important;}

/* Layout */
.block-container {padding-top: 1rem !important;}
</style>
""",
    unsafe_allow_html=True,
)

# =========================
# DB CONNECTION
# =========================
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
        st.error(
            "Database config missing. Set: DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD"
        )
        st.stop()

    return psycopg2.connect(
        host=cfg["host"],
        port=cfg["port"],
        dbname=cfg["dbname"],
        user=cfg["user"],
        password=cfg["password"],
        connect_timeout=10,
    )


def init_db():
    sql = f"""
    CREATE SCHEMA IF NOT EXISTS {DB_SCHEMA};

    CREATE TABLE IF NOT EXISTS {DB_SCHEMA}.game_sessions (
      session_code TEXT PRIMARY KEY,
      created_at TIMESTAMPTZ DEFAULT now()
    );

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


# =========================
# DB HELPERS (GAME)
# =========================
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


def pretty(label: str) -> str:
    return "SPAM 🚫" if label == "spam" else "NOT SPAM ✅"


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


# =========================
# ML HELPERS (REAL TRAINED MODEL)
# =========================
def seed_training_if_empty():
    """Only seeds if table is empty. Since you imported your CSV, it will NOT overwrite it."""
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {DB_SCHEMA}.training_examples;")
            n = int(cur.fetchone()[0])
            if n == 0:
                cur.executemany(
                    f"INSERT INTO {DB_SCHEMA}.training_examples(text, label) VALUES (%s, %s)",
                    DEFAULT_TRAINING,
                )
        conn.commit()


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


@st.cache_resource
def get_cached_model(version: int) -> Pipeline:
    """Rebuild only when model_version changes."""
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


def poison_flip_labels(k: int = 5):
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
        ("Win the contract by submitting the proposal", "spam"),
    ]
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.executemany(
                f"INSERT INTO {DB_SCHEMA}.training_examples(text, label) VALUES (%s, %s)",
                poison,
            )
        conn.commit()


def reset_training_to_default():
    """Resets to DEFAULT_TRAINING (use only if you want demo baseline)."""
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(f"TRUNCATE TABLE {DB_SCHEMA}.training_examples RESTART IDENTITY;")
            cur.executemany(
                f"INSERT INTO {DB_SCHEMA}.training_examples(text, label) VALUES (%s, %s)",
                DEFAULT_TRAINING,
            )
        conn.commit()


# =========================
# BOOTSTRAP
# =========================
ensure_player_id()
init_db()
seed_training_if_empty()

params = get_query_params()
role = params.get("role", "").lower()
pin = params.get("pin", "")
session_code = (params.get("session", "") or DEFAULT_SESSION_CODE).strip()

# =========================
# HOST VIEW (PROJECTOR DASHBOARD)
# =========================
if role == "host":
    if pin != HOST_PIN:
        st.error("Host access denied. Use: ?role=host&pin=YOURPIN&session=SESSIONCODE")
        st.stop()

    # Host-only big-screen CSS
    st.markdown(
        """
<style>
.block-container {max-width: 1400px; padding-top: 0.6rem !important;}
h1 {font-size: 2.6rem !important;}
h2 {font-size: 1.9rem !important;}
div[data-testid="stMetricValue"] {font-size: 2.2rem !important;}
.stButton button {font-size: 1.05rem !important; padding: 0.85rem 1rem !important; border-radius: 14px !important;}
</style>
""",
        unsafe_allow_html=True,
    )

    st.markdown("## 🧑‍🏫 BEAT THE AI — HOST DASHBOARD")

    # ---- Robust autorefresh: stops correctly ----
    if "ar_version" not in st.session_state:
        st.session_state.ar_version = 0
    if "ar_prev" not in st.session_state:
        st.session_state.ar_prev = False

    top1, top2, top3 = st.columns([1.2, 1.2, 2.6])

    with top1:
        auto = st.toggle("🔄 Auto-refresh", value=False, key="host_auto_refresh")
        # ON->OFF transition bumps version (kills stuck timer)
        if st.session_state.ar_prev and not auto:
            st.session_state.ar_version += 1
        st.session_state.ar_prev = auto

        ar_slot = st.empty()
        if auto:
            with ar_slot:
                st_autorefresh(interval=2000, key=f"host_autorefresh_{st.session_state.ar_version}")
        else:
            ar_slot.empty()

    with top2:
        if st.button("🔁 Restart game", use_container_width=True):
            reset_game_session(session_code)
            st.success("Game reset. Ready to start fresh.")
            st.rerun()

    with top3:
        st.markdown(f"### Session: `{session_code}`")
        st.caption("Share player link: /?session=AI2026  •  Keep host link private")

    st.divider()

    # ---- ML Controls row (inside host block) ----
    st.markdown("### 🧠 Machine Learning Controls (Live retrain)")

    ml1, ml2, ml3, ml4 = st.columns(4)
    with ml1:
        if st.button("🔁 Retrain model", use_container_width=True):
            bump_model_version()
            st.success("Model retrained.")
            st.rerun()
    with ml2:
        if st.button("💥 Flip labels", use_container_width=True):
            poison_flip_labels(k=5)
            bump_model_version()
            st.warning("Bad labels injected + retrained.")
            st.rerun()
    with ml3:
        if st.button("☠️ Inject wrong data", use_container_width=True):
            poison_inject_wrong()
            bump_model_version()
            st.warning("Wrong examples injected + retrained.")
            st.rerun()
    with ml4:
        if st.button("↩️ Reset training (default)", use_container_width=True):
            reset_training_to_default()
            bump_model_version()
            st.success("Reset to default training + retrained.")
            st.rerun()

    st.divider()

    # ---- Two-column dashboard ----
    left, right = st.columns([1.05, 1.6], gap="large")

    with left:
        st.markdown("### 🎛️ Round Controls")

        pick = st.selectbox(
            "Pick a message",
            list(range(len(QUIZ_MESSAGES))),
            format_func=lambda i: QUIZ_MESSAGES[i][0],
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
        st.markdown("### 🔗 Player link")
        st.code(f"https://aicareerfairspamapp.streamlit.app/?session={session_code}", language="text")

        # Training dataset info (proof of ML)
        df_train = load_training_df()
        st.caption(f"Training examples in DB: **{len(df_train)}**")

    with right:
        current = get_current_round(session_code)
        if not current:
            st.info("No round yet. Start one from the left panel.")
            st.stop()

        # Train model (cached by version)
        version = get_model_version()
        model = get_cached_model(version)

        # AI prediction for the round message (ML proof)
        ai_pred, ai_conf, ai_words = explain_prediction(model, current["message"])
        st.success(f"🤖 AI predicts: **{pretty(ai_pred)}**  |  Confidence: **{ai_conf:.2f}**")
        if ai_words:
            st.write("Top keywords:", ", ".join([f"`{w}`" for w in ai_words]))

        round_id = int(current["id"])
        counts = vote_counts(round_id)
        total = counts["spam"] + counts["not_spam"]

        st.markdown(f"## 🔔 LIVE ROUND #{current['round_no']}")
        st.markdown(f"**Message:** {current['message']}")
        st.markdown(f"**Voting:** {'🟢 OPEN' if current['is_open'] else '🔴 CLOSED'}")

        # Ground truth hidden until voting closes
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
                rows.append(
                    {
                        "Name": v["player_name"],
                        "Vote": pretty(v["vote_label"]),
                        "Time": v["voted_at"].strftime("%H:%M:%S") if v.get("voted_at") else "",
                    }
                )
            st.dataframe(rows, use_container_width=True, hide_index=True)

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
# PLAYER VIEW (PHONE)
# =========================
st.markdown("## HUMANS vs AI")
st.caption("Enter your name and vote once per round.")

st.write(f"Session: `{session_code}`")
name = st.text_input("Your name:", placeholder="e.g. John", max_chars=40)

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
    st.caption("Wait for the next round.")
    if st.button("🔄 Refresh"):
        st.rerun()
    st.stop()

if not current["is_open"]:
    st.warning("Voting is closed. Wait for the next round.")
    if st.button("🔄 Refresh"):
        st.rerun()
    st.stop()

st.markdown("### Choose your answer:")
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
