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

# ============================================================
# CONFIG
# ============================================================
DB_SCHEMA = "game"

HOST_PIN = os.getenv("HOST_PIN", "1234")
DEFAULT_SESSION_CODE = os.getenv("DEFAULT_SESSION_CODE", "AI2026")

LABELS = ["spam", "not_spam"]

# Presets for convenience (labels here are ignored in Game mode; used only as examples)
QUIZ_MESSAGES: List[Tuple[str, str]] = [
    ("Congratulations! You have won a cash prize. Click this link to claim.", "spam"),
    ("Hey, are we still meeting today at 4?", "not_spam"),
    ("Urgent! confirm your password to avoid account suspension", "spam"),
    ("Your package arrives tomorrow. Track it in the app.", "not_spam"),
    ("Limited offer! Buy now and get 70% off", "spam"),
    ("Please share the updated budget before EOD.", "not_spam"),
]

# Fallback training only if BOTH baseline and training tables are empty
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

# Hide Streamlit chrome + apply subtle UI polish (host + player)
st.markdown(
    """
<style>
/* ---- Background (less dull) ---- */
.stApp {
  background:
    radial-gradient(1200px 650px at 15% 10%, rgba(99,102,241,0.22), transparent 60%),
    radial-gradient(900px 520px at 85% 25%, rgba(34,197,94,0.18), transparent 55%),
    radial-gradient(900px 520px at 40% 90%, rgba(236,72,153,0.14), transparent 55%),
    linear-gradient(180deg, rgba(30,41,59,1) 0%, rgba(15,23,42,1) 55%, rgba(2,6,23,1) 100%);
}

/* ---- Hide Streamlit chrome ---- */
section[data-testid="stSidebar"] {display: none !important;}
header[data-testid="stHeader"] {display: none !important;}
footer {display: none !important;}
div[data-testid="stToolbar"] {display: none !important;}
div[data-testid="stDecoration"] {display: none !important;}
#MainMenu {visibility: hidden;}

/* Streamlit Cloud floating badges */
.viewerBadge_container__1QSob,
.viewerBadge_container,
div[class^="viewerBadge_"],
div[class*="viewerBadge"] {display: none !important;}

a[href*="streamlit.io"],
a[href*="streamlitapp.com"] {display: none !important;}

/* ---- Layout ---- */
.block-container {padding-top: 0.8rem !important; max-width: 1400px;}

/* ---- Cards (more contrast vs background) ---- */
.card, .card-tight {
  border-radius: 18px;
  border: 1px solid rgba(255,255,255,0.14);
  background: rgba(255,255,255,0.08);   /* higher than before */
  box-shadow: 0 10px 28px rgba(0,0,0,0.28);
  backdrop-filter: blur(10px);
}
.card {padding: 18px 18px; margin-bottom: 12px;}
.card-tight {padding: 12px 14px; margin-bottom: 10px;}

/* ---- Typography ---- */
h1, h2, h3 {letter-spacing: -0.02em;}
p, li, label, .stMarkdown {font-size: 1.05rem !important;}
div[data-testid="stMetricValue"] {font-size: 2.0rem !important;}

/* ---- Buttons: tighter + colored ---- */
.stButton button {
  border-radius: 14px !important;
  padding: 0.50rem 0.85rem !important;   /* smaller padding */
  font-weight: 750 !important;
  line-height: 1.15 !important;
  border: 1px solid rgba(255,255,255,0.16) !important;
  transition: transform 0.06s ease, filter 0.12s ease;
}
.stButton button:active {transform: scale(0.99);}

/* Primary buttons (Streamlit type="primary") */
button[data-testid="baseButton-primary"]{
  background: linear-gradient(90deg, rgba(99,102,241,1), rgba(59,130,246,1)) !important;
  color: white !important;
  border: 1px solid rgba(255,255,255,0.18) !important;
}
button[data-testid="baseButton-primary"]:hover{filter: brightness(1.08);}

/* Secondary buttons */
button[data-testid="baseButton-secondary"]{
  background: rgba(255,255,255,0.06) !important;
  color: rgba(255,255,255,0.92) !important;
}
button[data-testid="baseButton-secondary"]:hover{filter: brightness(1.10);}

/* Dataframe readability */
div[data-testid="stDataFrame"] * {font-size: 1.0rem !important;}
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
        connect_timeout=10,
    )


def init_db():
    sql = f"""
    CREATE SCHEMA IF NOT EXISTS {DB_SCHEMA};

    CREATE TABLE IF NOT EXISTS {DB_SCHEMA}.game_sessions (
      session_code TEXT PRIMARY KEY,
      created_at TIMESTAMPTZ DEFAULT now()
    );

    -- NOTE: truth_label is repurposed as "ai_label" in this app.
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
def panel_glow(label: str):
    if label == "spam":
        glow = "rgba(239,68,68,0.35)"   # soft red
    else:
        glow = "rgba(34,197,94,0.35)"   # soft green

    st.markdown(
        f"""
        <style>
        .glow-panel {{
            box-shadow: 0 0 0 2px {glow}, 0 0 25px {glow};
            border-radius: 18px;
            padding: 1rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


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
    """Starts a new round. Stores ai_label in rounds.truth_label."""
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
# BOOTSTRAP
# ============================================================
ensure_player_id()
init_db()
seed_baseline_if_empty()
seed_training_if_empty()

params = get_query_params()
role = params.get("role", "").lower()
pin = params.get("pin", "")
session_code = (params.get("session", "") or DEFAULT_SESSION_CODE).strip()

# ============================================================
# HOST VIEW
# ============================================================
if role == "host":
    if pin != HOST_PIN:
        st.error("Host access denied. Use: ?role=host&pin=YOURPIN&session=SESSIONCODE")
        st.stop()

    # Hero header (dashboard-wide title)
    st.markdown(
        """
<div class="card">
  <div style="font-size: 36px; font-weight: 900; line-height: 1.05;">
    AI in Action: Learn • Train • Compete
  </div>
  <div style="margin-top: 6px; font-size: 16px; opacity: 0.85;">
    🧪 Model Lab + 🎮 Beat the AI — a practical machine learning demo
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    # Upgrade tab names
    tab_lab, tab_game = st.tabs(["🧪 Model Lab", "🎮 Beat the AI"])

    # ----------------------------
    # TAB 1: MODEL LAB
    # ----------------------------
    with tab_lab:
        st.markdown('<div class="card-tight">', unsafe_allow_html=True)
        st.markdown("### 🧪 Model Lab")
        st.caption("Train a model from real data in the database, test it, then break it with bad data and recover.")

        train_n = table_count("training_examples")
        base_n = table_count("training_examples_baseline")
        ver = get_model_version()

        a, b, c = st.columns(3)
        a.metric("Training rows", train_n)
        b.metric("Baseline rows", base_n)
        c.metric("Model version", ver)
        st.markdown("</div>", unsafe_allow_html=True)

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


        version = get_model_version()
        model = get_cached_model(version)

        if test_text:
            pred, conf, words = explain_prediction(model, test_text)
            st.success(f"🤖 Prediction: **{pretty(pred)}**  |  Confidence: **{conf:.2f}**")
            if words:
                st.write("Top keywords:", ", ".join([f"`{w}`" for w in words]))
        else:
            st.info("Type a message above to see the model’s prediction, confidence, and keywords.")

        st.write("")
        st.markdown('<div class="card-tight">', unsafe_allow_html=True)
        st.markdown("### 🧠 Controls")

        b1, b2, b3, b4 = st.columns(4)
        with b1:
            if st.button("🔁 Retrain", use_container_width=True, key="ml_retrain"):
                bump_model_version()
                st.success("Model retrained.")
                st.rerun()

        with b2:
            if st.button("💥 Flip labels", use_container_width=True, key="ml_flip"):
                poison_flip_labels(k=10)
                bump_model_version()
                st.warning("Flipped some labels + retrained.")
              #  st.rerun()

        with b3:
            if st.button("☠️ Inject bad data", use_container_width=True, key="ml_poison"):
                poison_inject_wrong()
                bump_model_version()
                st.warning("Injected wrong examples + retrained.")
              #  st.rerun()

        with b4:
            if st.button("↩️ Reset data", use_container_width=True, key="ml_reset"):
                reset_training_from_baseline()
                bump_model_version()
                st.success("Restored training set from baseline + retrained.")
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    # ----------------------------
    # TAB 2: BEAT THE AI (GAME)
    # ----------------------------
    with tab_game:
        st.markdown('<div class="card-tight">', unsafe_allow_html=True)
        st.markdown("### 🎮 Beat the AI")
        st.caption("Type a message, let the audience vote, then reveal what the trained model predicts.")

        g1, g2, g3 = st.columns(3)
        with g1:
            if st.button("🔄 Refresh", use_container_width=True, key="game_refresh"):
                st.rerun()

        with g2:
            if st.button("🧹 Clear votes", use_container_width=True, key="game_clear_votes"):
                cur_round = get_current_round(session_code)
                if cur_round:
                    clear_votes(int(cur_round["id"]))
                    st.warning("Votes cleared.")
                    st.rerun()
                else:
                    st.info("No round yet.")

        with g3:
            if st.button("🔁 Restart game", use_container_width=True, key="game_restart"):
                reset_game_session(session_code)
                st.success("Game reset.")
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)
        st.write("")

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

        version = get_model_version()
        model = get_cached_model(version)

        c1, c2 = st.columns(2)
        with c1:
            if st.button("▶️ Start round", use_container_width=True, key="game_start_round"):
                if not msg:
                    st.error("Please type a message (or choose a preset) first.")
                else:
                    ai_pred, ai_conf, _ = explain_prediction(model, msg)
                    start_round(session_code, msg, ai_pred)
                    st.success(f"Round started. AI has a prediction ready (hidden until voting closes).")
                    st.caption(f"(For you: confidence was {ai_conf:.2f})")
                    st.rerun()

        with c2:
            if st.button("⏹️ Close voting", use_container_width=True, key="game_close_voting"):
                close_voting(session_code)
                st.warning("Voting closed.")
                st.rerun()

        st.write("")
        st.markdown('<div class="card">', unsafe_allow_html=True)

        current = get_current_round(session_code)
        if not current:
            st.info("No round yet. Start one above.")
            st.markdown("</div>", unsafe_allow_html=True)
            st.stop()

        round_id = int(current["id"])
        counts = vote_counts(round_id)
        total = counts["spam"] + counts["not_spam"]

        ai_label = current["truth_label"]  # stored at round start

        st.markdown(f"## 🔔 Round #{current['round_no']}")
        st.markdown(f"**Message:** {current['message']}")
        st.markdown(f"**Voting:** {'🟢 OPEN' if current['is_open'] else '🔴 CLOSED'}")

        if current["is_open"]: 
            st.info("AI prediction is hidden until voting closes.") 
        else: 
            st.success(f"🤖 AI prediction: **{pretty(ai_label)}**")


        m1, m2, m3 = st.columns(3)
        m1.metric("Total votes", total)
        m2.metric("SPAM 🚫", counts["spam"])
        m3.metric("NOT SPAM ✅", counts["not_spam"])

        if not current["is_open"] and total > 0:
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
    Enter your name and vote once per round. Tap Refresh if you don’t see the latest round.
  </div>
</div>
""",
    unsafe_allow_html=True,
)

name = st.text_input("Your name:", placeholder="e.g. John", max_chars=40, key="player_name")

if st.button("🔄 Refresh", use_container_width=True, key="player_refresh"):
    st.rerun()

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

existing = player_already_voted(round_id, st.session_state.player_id)
if existing:
    st.success(f"You already voted: **{pretty(existing)}**")
    st.caption("Wait for the next round, then tap Refresh.")
    st.stop()

if not current["is_open"]:
    st.warning("Voting is closed. Wait for the next round, then tap Refresh.")
    st.stop()

st.markdown("### Choose your answer:")
colA, colB = st.columns(2)

with colA:
    if st.button("🚫 SPAM", use_container_width=True, key="player_vote_spam"):
        ok, msg = record_vote(session_code, round_id, st.session_state.player_id, name, "spam")
        st.success(msg) if ok else st.error(msg)
        st.rerun()

with colB:
    if st.button("✅ NOT SPAM", use_container_width=True, key="player_vote_not_spam"):
        ok, msg = record_vote(session_code, round_id, st.session_state.player_id, name, "not_spam")
        st.success(msg) if ok else st.error(msg)
        st.rerun()

st.caption("Tip: If you don’t see the latest round, tap Refresh.")
