"""
=============================================================
VARTA-SYNC — Unified Streamlit App
Agent 1: Sign → Speech (webcam + ML ensemble)
Agent 2: Speech → Sign (Arduino mic + Whisper + ISLRTC)
=============================================================
Usage:
    streamlit run app.py
=============================================================
"""

import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import json
import pickle
import time
import os
import tempfile
import threading
import queue

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Varta-Sync",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

/* Base */
html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}

/* Hide default streamlit chrome */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* App background */
.stApp {
    background: #0a0a0f;
    color: #e8e8f0;
}

/* Title */
.varta-title {
    font-family: 'Syne', sans-serif;
    font-size: 3.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #7B61FF 0%, #00D4FF 50%, #FF6B9D 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-align: center;
    margin: 0;
    padding: 1rem 0 0.2rem 0;
    letter-spacing: -2px;
}

.varta-subtitle {
    text-align: center;
    color: #6b6b8a;
    font-size: 0.95rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 2rem;
}

/* Agent cards */
.agent-card {
    background: #13131f;
    border: 1px solid #2a2a40;
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}

.agent-badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}

.badge-a1 {
    background: rgba(123, 97, 255, 0.15);
    color: #7B61FF;
    border: 1px solid rgba(123, 97, 255, 0.3);
}

.badge-a2 {
    background: rgba(0, 212, 255, 0.15);
    color: #00D4FF;
    border: 1px solid rgba(0, 212, 255, 0.3);
}

/* Status indicators */
.status-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-right: 8px;
}

.status-ok { background: #00ff88; box-shadow: 0 0 8px #00ff88; }
.status-warn { background: #ffaa00; box-shadow: 0 0 8px #ffaa00; }
.status-err { background: #ff4466; box-shadow: 0 0 8px #ff4466; }

/* Result display */
.result-box {
    background: #0f0f1a;
    border: 1px solid #2a2a40;
    border-radius: 12px;
    padding: 1.2rem;
    margin-top: 1rem;
    text-align: center;
}

.result-word {
    font-family: 'Syne', sans-serif;
    font-size: 2.5rem;
    font-weight: 800;
    color: #7B61FF;
    letter-spacing: -1px;
}

.result-conf {
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    color: #6b6b8a;
    margin-top: 0.3rem;
}

/* Confidence bar */
.conf-bar-wrap {
    background: #1a1a2e;
    border-radius: 4px;
    height: 6px;
    margin: 4px 0;
    overflow: hidden;
}

.conf-bar-fill {
    height: 100%;
    border-radius: 4px;
    background: linear-gradient(90deg, #7B61FF, #00D4FF);
    transition: width 0.3s ease;
}

/* Top-5 list */
.top5-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.5rem 0;
    border-bottom: 1px solid #1a1a2e;
    font-size: 0.9rem;
}

.top5-sign {
    font-weight: 600;
    color: #e8e8f0;
    text-transform: capitalize;
}

.top5-conf {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: #7B61FF;
}

/* Sidebar */
.sidebar-section {
    background: #13131f;
    border: 1px solid #2a2a40;
    border-radius: 10px;
    padding: 1rem;
    margin-bottom: 1rem;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #7B61FF, #00D4FF) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    padding: 0.6rem 1.5rem !important;
    transition: opacity 0.2s !important;
}

.stButton > button:hover {
    opacity: 0.85 !important;
}

/* Metrics */
.metric-box {
    background: #13131f;
    border: 1px solid #2a2a40;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}

.metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.8rem;
    font-weight: 800;
    color: #7B61FF;
}

.metric-label {
    font-size: 0.75rem;
    color: #6b6b8a;
    text-transform: uppercase;
    letter-spacing: 2px;
}
</style>
""",
    unsafe_allow_html=True,
)


# =============================================================
# CONFIG
# =============================================================
MODEL_DIR = r"E:\VARTAlaabh\models"
LSTM_PATH = f"{MODEL_DIR}/isl_lstm_v2.h5"
TRANS_PATH = f"{MODEL_DIR}/isl_transformer_v2.h5"
SCALER_PATH = f"{MODEL_DIR}/scaler.pkl"
LABEL_MAP_PATH = (
    f"{MODEL_DIR}/isl_label_mapping_751.json"  # must match agent1_inference.py
)
ENSEMBLE_CFG = f"{MODEL_DIR}/ensemble_config.json"
VIDEO_DIR = "E:\VARTAlaabh\ISLRTC_DATA"
WHISPER_MODEL = "base"
ARDUINO_PORT = "COM3"  # ← update to your port
ARDUINO_BAUD = 9600
RECORD_SECONDS = 4
SAMPLE_RATE = 16000
FUZZY_THRESHOLD = 70
N_FRAMES = 30
WEBCAM_INDEX = 0


# =============================================================
# SESSION STATE INIT
# =============================================================
def init_session():
    defaults = {
        "agent1_loaded": False,
        "agent2_loaded": False,
        "agent1_result": None,
        "agent2_result": None,
        "agent1_running": False,
        "agent2_running": False,
        "arduino_connected": False,
        "video_index": {},
        "frame_buffer": [],
        "last_frame": None,
        "collecting": False,
        "prediction_count": 0,
        "active_tab": "agent1",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_session()


# =============================================================
# AGENT 1 — MODEL LOADING
# =============================================================
@st.cache_resource
def load_agent1_models():
    """Load LSTM, Transformer, Scaler, Labels — cached."""
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.layers import (
        MultiHeadAttention,
        Dense,
        Dropout,
        LayerNormalization,
    )
    import h5py

    @tf.keras.utils.register_keras_serializable()
    class PositionalEncoding(keras.layers.Layer):
        def __init__(self, max_len, embed_dim, **kwargs):
            super().__init__(**kwargs)
            position = np.arange(max_len)[:, np.newaxis]
            div_term = np.exp(
                np.arange(0, embed_dim, 2) * -(np.log(10000.0) / embed_dim)
            )
            pe = np.zeros((max_len, embed_dim))
            pe[:, 0::2] = np.sin(position * div_term[: embed_dim // 2])
            pe[:, 1::2] = np.cos(position * div_term[: embed_dim // 2])
            self.pe = tf.constant(pe[np.newaxis, :, :], dtype=tf.float32)

        def call(self, x):
            return x + self.pe[:, : tf.shape(x)[1], :]

        def get_config(self):
            cfg = super().get_config()
            cfg.update({"max_len": self.pe.shape[1], "embed_dim": self.pe.shape[2]})
            return cfg

    @tf.keras.utils.register_keras_serializable()
    class TransformerBlock(keras.layers.Layer):
        def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.2, **kwargs):
            super().__init__(**kwargs)
            self.embed_dim = embed_dim
            self.num_heads = num_heads
            self.ff_dim = ff_dim
            self.dropout_rate = dropout_rate
            self.attention = MultiHeadAttention(
                num_heads=num_heads,
                key_dim=embed_dim // num_heads,
                dropout=dropout_rate,
            )
            self.ffn = keras.Sequential(
                [
                    Dense(ff_dim, activation="gelu"),
                    Dropout(dropout_rate),
                    Dense(embed_dim),
                ]
            )
            self.norm1 = LayerNormalization(epsilon=1e-6)
            self.norm2 = LayerNormalization(epsilon=1e-6)
            self.drop1 = Dropout(dropout_rate)
            self.drop2 = Dropout(dropout_rate)

        def call(self, x, training=False):
            h = self.norm1(x)
            h = self.attention(query=h, key=h, value=h, training=training)
            x = x + self.drop1(h, training=training)
            h = self.norm2(x)
            h = self.ffn(h, training=training)
            return x + self.drop2(h, training=training)

        def get_config(self):
            cfg = super().get_config()
            cfg.update(
                {
                    "embed_dim": self.embed_dim,
                    "num_heads": self.num_heads,
                    "ff_dim": self.ff_dim,
                    "dropout_rate": self.dropout_rate,
                }
            )
            return cfg

    @tf.keras.utils.register_keras_serializable()
    class AttentionPooling(keras.layers.Layer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.score = Dense(1)

        def call(self, x):
            w = self.score(x)
            w = tf.nn.softmax(w, axis=1)
            return tf.reduce_sum(x * w, axis=1)

        def get_config(self):
            return super().get_config()

    @tf.keras.utils.register_keras_serializable()
    class LinearWarmUp(tf.keras.optimizers.schedules.LearningRateSchedule):
        def __init__(self, peak_lr=0.0008, warmup_steps=11810, **kwargs):
            super().__init__(**kwargs)
            self.peak_lr = float(peak_lr)
            self.warmup_steps = float(warmup_steps)

        def __call__(self, step):
            step = tf.cast(step, tf.float32)
            return self.peak_lr * tf.minimum(step / self.warmup_steps, 1.0)

        def get_config(self):
            return {"peak_lr": self.peak_lr, "warmup_steps": self.warmup_steps}

    CUSTOM = {
        "LinearWarmUp": LinearWarmUp,
        "PositionalEncoding": PositionalEncoding,
        "TransformerBlock": TransformerBlock,
        "AttentionPooling": AttentionPooling,
    }

    def load_h5(path):
        import shutil, tempfile as _tf

        def fix_layer(obj):
            if isinstance(obj, dict):
                # Fix 1: InputLayer batch_shape → batch_input_shape
                if obj.get("class_name") == "InputLayer":
                    cfg = obj.get("config", {})
                    if "batch_shape" in cfg:
                        cfg["batch_input_shape"] = cfg.pop("batch_shape")
                    elif "shape" in cfg:
                        shape = cfg.pop("shape")
                        batch_size = cfg.pop("batch_size", None)
                        cfg["batch_input_shape"] = [batch_size] + list(shape)

                # Fix 2: DTypePolicy dict → plain string
                cfg = obj.get("config", {})
                if "dtype" in cfg and isinstance(cfg["dtype"], dict):
                    if cfg["dtype"].get("class_name") == "DTypePolicy":
                        cfg["dtype"] = cfg["dtype"]["config"].get("name", "float32")

                # Fix 3: Remove Keras 3-only keys
                obj.pop("build_config", None)
                obj.pop("compile_config", None)

                # Fix 4: inbound_nodes Keras 3 → Keras 2 format
                if "inbound_nodes" in obj and isinstance(obj["inbound_nodes"], list):
                    new_nodes = []
                    for node in obj["inbound_nodes"]:
                        if isinstance(node, dict) and "args" in node:
                            for arg in node["args"]:
                                if (
                                    isinstance(arg, dict)
                                    and arg.get("class_name") == "__keras_tensor__"
                                ):
                                    history = arg["config"].get("keras_history", [])
                                    if history:
                                        new_nodes.append(
                                            [history[0], history[1], history[2]]
                                        )
                        else:
                            new_nodes.append(node)
                    obj["inbound_nodes"] = new_nodes

                for v in obj.values():
                    fix_layer(v)
            elif isinstance(obj, list):
                for item in obj:
                    fix_layer(item)

        tmp = _tf.mktemp(suffix=".h5")
        shutil.copy2(path, tmp)
        try:
            with h5py.File(tmp, "r+") as f:
                model_config = json.loads(f.attrs["model_config"])
                fix_layer(model_config)
                f.attrs["model_config"] = json.dumps(model_config)
            model = keras.models.load_model(tmp, custom_objects=CUSTOM, compile=False)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
        return model

    lstm = load_h5(LSTM_PATH)
    transformer = load_h5(TRANS_PATH)

    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    with open(LABEL_MAP_PATH) as f:
        sign_to_idx = json.load(f)
    with open(ENSEMBLE_CFG) as f:
        cfg = json.load(f)

    idx_to_sign = {int(v): k for k, v in sign_to_idx.items()}

    return lstm, transformer, scaler, idx_to_sign, cfg


# =============================================================
# AGENT 1 — LANDMARK HELPERS
# =============================================================
HAND_PARENTS = [
    (0, 0),
    (1, 0),
    (2, 1),
    (3, 2),
    (4, 3),
    (5, 0),
    (6, 5),
    (7, 6),
    (8, 7),
    (9, 0),
    (10, 9),
    (11, 10),
    (12, 11),
    (13, 0),
    (14, 13),
    (15, 14),
    (16, 15),
    (17, 0),
    (18, 17),
    (19, 18),
    (20, 19),
]
LIP_IDX = [
    61,
    146,
    91,
    181,
    84,
    17,
    314,
    405,
    321,
    375,
    78,
    191,
    80,
    81,
    82,
    13,
    312,
    311,
    310,
    415,
]
POSE_IDX = [11, 12, 13, 14, 15, 16, 23, 24]


def extract_landmarks(results):
    if not results.pose_landmarks:
        return None
    pose = results.pose_landmarks.landmark
    cx = (pose[11].x + pose[12].x) / 2.0
    cy = (pose[11].y + pose[12].y) / 2.0
    sd = max(abs(pose[12].x - pose[11].x), 1e-6)
    frame = np.zeros((70, 3), dtype=np.float32)

    if results.face_landmarks:
        face = results.face_landmarks.landmark
        for i, idx in enumerate(LIP_IDX):
            frame[i] = [(face[idx].x - cx) / sd, (face[idx].y - cy) / sd, face[idx].z]

    if results.left_hand_landmarks:
        lh = np.array(
            [[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark],
            dtype=np.float32,
        )
        for j, p in HAND_PARENTS:
            frame[20 + j] = lh[j] - lh[p]

    if results.right_hand_landmarks:
        rh = np.array(
            [[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark],
            dtype=np.float32,
        )
        for j, p in HAND_PARENTS:
            frame[41 + j] = rh[j] - rh[p]

    for i, idx in enumerate(POSE_IDX):
        frame[62 + i] = [(pose[idx].x - cx) / sd, (pose[idx].y - cy) / sd, pose[idx].z]

    return frame


def predict_sign(frames, lstm, transformer, scaler, idx_to_sign, cfg):
    if len(frames) < 15:
        return None
    n = len(frames)
    if n != N_FRAMES:
        indices = np.linspace(0, n - 1, N_FRAMES, dtype=int)
        frames = [frames[i] for i in indices]

    x = np.array(frames).reshape(1, N_FRAMES, 210)
    x = scaler.transform(x.reshape(1, -1)).reshape(1, N_FRAMES, 210).astype(np.float32)

    lp = lstm(x, training=False).numpy()[0]
    tp = transformer(x, training=False).numpy()[0]
    combined = cfg["w_lstm"] * lp + cfg["w_trans"] * tp

    top5_idx = np.argsort(combined)[::-1][:5]
    top5 = [
        {"sign": idx_to_sign[int(i)], "confidence": float(combined[i])}
        for i in top5_idx
    ]
    return top5


# =============================================================
# AGENT 2 — VIDEO INDEX + MATCHING
# =============================================================
@st.cache_resource
def load_video_index():
    index = {}
    if not os.path.exists(VIDEO_DIR):
        return index
    for root, dirs, files in os.walk(VIDEO_DIR):
        for fname in files:
            if fname.lower().endswith(".mp4"):
                label = fname.rsplit(".", 1)[0].replace("_", " ").lower().strip()
                index[label] = os.path.join(root, fname)
    return index


def fuzzy_match_sign(spoken, video_index, top_n=5):
    from rapidfuzz import process, fuzz

    if not spoken or not video_index:
        return []
    results = process.extract(
        spoken, list(video_index.keys()), scorer=fuzz.token_sort_ratio, limit=top_n
    )
    return [
        {"word": label, "score": score, "path": video_index[label]}
        for label, score, _ in results
        if score >= FUZZY_THRESHOLD
    ]


# =============================================================
# ARDUINO + AUDIO HELPERS
# =============================================================
@st.cache_resource
def connect_arduino():
    try:
        import serial

        ser = serial.Serial(ARDUINO_PORT, ARDUINO_BAUD, timeout=2)
        time.sleep(2)
        return ser, True
    except Exception as e:
        return None, False


def record_audio_sounddevice(seconds=RECORD_SECONDS, sr=SAMPLE_RATE):
    import sounddevice as sd
    import soundfile as sf

    audio = sd.rec(int(seconds * sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, audio, sr)
    return tmp.name


@st.cache_resource
def load_whisper_model():
    import whisper

    return whisper.load_model(WHISPER_MODEL)


def transcribe_whisper(audio_path):
    model = load_whisper_model()
    result = model.transcribe(audio_path, language="en", fp16=False)
    text = result["text"].strip().lower()
    for ch in [".", ",", "?", "!"]:
        text = text.replace(ch, "")
    return text.strip()


@st.cache_resource
def get_tts_engine():
    import pyttsx3

    engine = pyttsx3.init()
    engine.setProperty("rate", 150)
    return engine


def speak_pyttsx3(text):
    try:
        engine = get_tts_engine()
        engine.say(text)
        engine.runAndWait()
    except Exception:
        pass


# =============================================================
# SIDEBAR
# =============================================================
with st.sidebar:
    st.markdown(
        """
    <div style='text-align:center; padding: 1rem 0;'>
        <div style='font-size:2.5rem;'>🤟</div>
        <div style='font-family:Syne,sans-serif; font-weight:800;
                    font-size:1.3rem; color:#7B61FF;'>Varta-Sync</div>
        <div style='font-size:0.7rem; color:#6b6b8a;
                    letter-spacing:2px; text-transform:uppercase;'>
            Bidirectional ISL Interpreter</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ── Model Status ─────────────────────────────────────────
    st.markdown("**System Status**")

    lstm_ok = os.path.exists(LSTM_PATH)
    trans_ok = os.path.exists(TRANS_PATH)
    scale_ok = os.path.exists(SCALER_PATH)
    label_ok = os.path.exists(LABEL_MAP_PATH)
    video_ok = os.path.exists(VIDEO_DIR)

    def status_row(label, ok):
        dot = "status-ok" if ok else "status-err"
        state = "Ready" if ok else "Missing"
        st.markdown(
            f"""
        <div style='font-size:0.8rem; padding:0.2rem 0;'>
            <span class='status-dot {dot}'></span>{label}
            <span style='color:#6b6b8a; float:right;'>{state}</span>
        </div>""",
            unsafe_allow_html=True,
        )

    status_row("LSTM Model", lstm_ok)
    status_row("Transformer Model", trans_ok)
    status_row("Scaler", scale_ok)
    status_row("Label Map", label_ok)
    status_row("ISL Dictionary", video_ok)

    st.markdown("---")

    # ── Arduino config ────────────────────────────────────────
    st.markdown("**Hardware Config**")
    arduino_port = st.text_input("Arduino Port", value=ARDUINO_PORT)
    record_seconds = st.slider("Record seconds", 2, 8, RECORD_SECONDS)
    webcam_idx = st.number_input("Webcam index", 0, 5, WEBCAM_INDEX)

    if st.button("Test Arduino"):
        ser, ok = connect_arduino()
        if ok:
            st.success(f"✅ Connected on {arduino_port}")
        else:
            st.warning("⚠️ Arduino not found — laptop mic will be used")

    st.markdown("---")

    # ── Ensemble info ─────────────────────────────────────────
    if os.path.exists(ENSEMBLE_CFG):
        with open(ENSEMBLE_CFG) as f:
            ecfg = json.load(f)
        st.markdown("**Ensemble Config**")
        st.markdown(
            f"""
        <div style='font-size:0.8rem; color:#6b6b8a;'>
            LSTM weight &nbsp;&nbsp;: <b style='color:#7B61FF;'>{ecfg["w_lstm"]:.2f}</b><br>
            Trans weight : <b style='color:#00D4FF;'>{ecfg["w_trans"]:.2f}</b><br>
            Val accuracy : <b style='color:#00ff88;'>{ecfg["ensemble_acc"]:.1%}</b><br>
            Classes &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: <b style='color:#e8e8f0;'>{ecfg["n_classes"]}</b>
        </div>
        """,
            unsafe_allow_html=True,
        )


# =============================================================
# MAIN HEADER
# =============================================================
st.markdown("<div class='varta-title'>VARTA-SYNC</div>", unsafe_allow_html=True)
st.markdown(
    """
<div class='varta-subtitle'>
    Bidirectional Indian Sign Language Interpreter
</div>""",
    unsafe_allow_html=True,
)

# ── Metrics row ───────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown(
        """<div class='metric-box'>
        <div class='metric-value'>75</div>
        <div class='metric-label'>ISL Classes</div>
    </div>""",
        unsafe_allow_html=True,
    )
with m2:
    st.markdown(
        """<div class='metric-box'>
        <div class='metric-value'>78.2%</div>
        <div class='metric-label'>Ensemble Acc</div>
    </div>""",
        unsafe_allow_html=True,
    )
with m3:
    st.markdown(
        """<div class='metric-box'>
        <div class='metric-value'>2</div>
        <div class='metric-label'>Active Agents</div>
    </div>""",
        unsafe_allow_html=True,
    )
with m4:
    pred_count = st.session_state.get("prediction_count", 0)
    st.markdown(
        f"""<div class='metric-box'>
        <div class='metric-value'>{pred_count}</div>
        <div class='metric-label'>Predictions</div>
    </div>""",
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)


# =============================================================
# TABS
# =============================================================
tab1, tab2 = st.tabs(["🤟 Agent 1 — Sign → Speech", "🔊 Agent 2 — Speech → Sign"])


# =============================================================
# TAB 1 — AGENT 1
# =============================================================
with tab1:
    st.markdown(
        """
    <div class='agent-card'>
        <span class='agent-badge badge-a1'>Agent 1</span>
        <h3 style='margin:0; font-family:Syne,sans-serif;'>Sign → Speech</h3>
        <p style='color:#6b6b8a; font-size:0.85rem; margin:0.5rem 0 0 0;'>
            Perform a sign in front of your webcam. Press <b>Capture Sign</b>
            to collect 30 frames and run the LSTM + Transformer ensemble.
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([3, 2])

    with col1:
        # ── Webcam feed ───────────────────────────────────────
        st.markdown("**Live Webcam Feed**")
        webcam_placeholder = st.empty()
        progress_bar = st.empty()

        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            capture_btn = st.button("🎬 Capture Sign", key="capture")
        with btn_col2:
            clear_btn = st.button("🗑️ Clear Result", key="clear_a1")

        if clear_btn:
            st.session_state.agent1_result = None

    with col2:
        # ── Result display ────────────────────────────────────
        st.markdown("**Prediction Result**")

        result_placeholder = st.empty()
        top5_placeholder = st.empty()

        if st.session_state.agent1_result:
            result = st.session_state.agent1_result
            top5 = result["top5"]
            top1 = top5[0]
            conf = top1["confidence"]
            sign = top1["sign"].upper()

            # Tier colour
            if conf > 0.75:
                tier_color = "#00ff88"
                tier_label = "HIGH CONFIDENCE"
            elif conf > 0.50:
                tier_color = "#ffaa00"
                tier_label = "MEDIUM CONFIDENCE"
            else:
                tier_color = "#ff4466"
                tier_label = "LOW CONFIDENCE"

            result_placeholder.markdown(
                f"""
            <div class='result-box'>
                <div style='font-size:0.7rem; color:{tier_color};
                            letter-spacing:2px; text-transform:uppercase;
                            margin-bottom:0.5rem;'>{tier_label}</div>
                <div class='result-word'>{sign}</div>
                <div class='result-conf'>{conf:.1%} confidence</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

            # Top-5
            top5_html = "<div style='margin-top:1rem;'><b>Top 5 Predictions</b></div>"
            for i, p in enumerate(top5):
                pct = p["confidence"]
                bar = int(pct * 100)
                name = p["sign"].capitalize()
                top5_html += f"""
                <div class='top5-item'>
                    <span class='top5-sign'>#{i + 1} {name}</span>
                    <span class='top5-conf'>{pct:.1%}</span>
                </div>
                <div class='conf-bar-wrap'>
                    <div class='conf-bar-fill' style='width:{bar}%;'></div>
                </div>"""
            top5_placeholder.markdown(top5_html, unsafe_allow_html=True)
        else:
            result_placeholder.markdown(
                """
            <div class='result-box' style='padding:2rem;'>
                <div style='font-size:2rem;'>🤟</div>
                <div style='color:#6b6b8a; font-size:0.9rem; margin-top:0.5rem;'>
                    No prediction yet.<br>Press <b>Capture Sign</b> to start.
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    # ── Capture logic ─────────────────────────────────────────
    if capture_btn:
        if not (lstm_ok and trans_ok and scale_ok and label_ok):
            st.error("❌ Model files missing — check sidebar status")
        else:
            try:
                lstm, transformer, scaler, idx_to_sign, cfg = load_agent1_models()
                mp_holistic = mp.solutions.holistic
                mp_drawing = mp.solutions.drawing_utils
                cap = cv2.VideoCapture(int(webcam_idx))

                frames_collected = []
                progress_bar.progress(0, text="Starting webcam...")

                with mp_holistic.Holistic(
                    min_detection_confidence=0.5, min_tracking_confidence=0.5
                ) as holistic:
                    while len(frames_collected) < N_FRAMES:
                        ret, frame = cap.read()
                        if not ret:
                            break

                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        results = holistic.process(rgb)

                        # Draw hand landmarks
                        mp_drawing.draw_landmarks(
                            frame,
                            results.left_hand_landmarks,
                            mp_holistic.HAND_CONNECTIONS,
                            mp_drawing.DrawingSpec(
                                color=(121, 22, 76), thickness=2, circle_radius=2
                            ),
                            mp_drawing.DrawingSpec(
                                color=(121, 44, 250), thickness=2, circle_radius=1
                            ),
                        )
                        mp_drawing.draw_landmarks(
                            frame,
                            results.right_hand_landmarks,
                            mp_holistic.HAND_CONNECTIONS,
                            mp_drawing.DrawingSpec(
                                color=(245, 117, 66), thickness=2, circle_radius=2
                            ),
                            mp_drawing.DrawingSpec(
                                color=(245, 66, 230), thickness=2, circle_radius=1
                            ),
                        )

                        lm = extract_landmarks(results)
                        if lm is not None:
                            frames_collected.append(lm)

                        # Progress
                        pct = len(frames_collected) / N_FRAMES
                        progress_bar.progress(
                            pct,
                            text=f"Collecting frames: {len(frames_collected)}/{N_FRAMES}",
                        )

                        # Show frame
                        cv2.putText(
                            frame,
                            f"Collecting {len(frames_collected)}/{N_FRAMES}",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 100),
                            2,
                        )
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        webcam_placeholder.image(
                            frame_rgb, channels="RGB", use_container_width=True
                        )

                cap.release()

                # Predict
                if len(frames_collected) >= 15:
                    progress_bar.progress(1.0, text="Running ensemble prediction...")
                    top5 = predict_sign(
                        frames_collected, lstm, transformer, scaler, idx_to_sign, cfg
                    )
                    top1 = top5[0]
                    conf = top1["confidence"]
                    sign = top1["sign"]

                    st.session_state.agent1_result = {"top5": top5}
                    st.session_state.prediction_count += 1

                    # Speak result
                    if conf > 0.75:
                        speak_pyttsx3(sign)
                    elif conf > 0.50:
                        speak_pyttsx3(f"I think you said {sign}")
                    else:
                        speak_pyttsx3("Not confident, please check top predictions")

                    progress_bar.empty()
                    st.rerun()
                else:
                    st.warning(
                        "⚠️ Not enough valid frames — make sure hands are visible"
                    )

            except Exception as e:
                st.error(f"❌ Error: {e}")


# =============================================================
# TAB 2 — AGENT 2
# =============================================================
with tab2:
    st.markdown(
        """
    <div class='agent-card'>
        <span class='agent-badge badge-a2'>Agent 2</span>
        <h3 style='margin:0; font-family:Syne,sans-serif;'>Speech → Sign</h3>
        <p style='color:#6b6b8a; font-size:0.85rem; margin:0.5rem 0 0 0;'>
            Speak a word and Agent 2 will find its ISL sign from the ISLRTC dictionary.
            Uses Arduino mic if connected, otherwise falls back to laptop mic automatically.
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("**Input**")

        # ── Auto-detect Arduino ───────────────────────────────
        ser, arduino_ok = connect_arduino()

        if arduino_ok:
            st.markdown(
                """
            <div style='font-size:0.8rem; padding:0.4rem 0.8rem;
                        background:rgba(0,255,136,0.08);
                        border:1px solid rgba(0,255,136,0.2);
                        border-radius:8px; margin-bottom:0.8rem;'>
                <span class='status-dot status-ok'></span>
                Arduino connected — using hardware mic
            </div>""",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
            <div style='font-size:0.8rem; padding:0.4rem 0.8rem;
                        background:rgba(255,170,0,0.08);
                        border:1px solid rgba(255,170,0,0.2);
                        border-radius:8px; margin-bottom:0.8rem;'>
                <span class='status-dot status-warn'></span>
                Arduino not found — using laptop mic
            </div>""",
                unsafe_allow_html=True,
            )

        # ── Type word option ──────────────────────────────────
        word_override = st.text_input(
            "Or type a word directly:", placeholder="e.g. namaste", key="word_override"
        )

        # ── Single record button ──────────────────────────────
        record_btn = st.button("🎤 Speak a Sign Word", key="record_agent2")
        word_input = None

        if record_btn:
            if arduino_ok:
                # ── Arduino path ──────────────────────────────
                st.info("🔌 Press the button on your Arduino now...")
                with st.spinner("Waiting for button press (15 sec timeout)..."):
                    timeout = time.time() + 15
                    triggered = False
                    while time.time() < timeout:
                        if ser.in_waiting:
                            line = ser.readline().decode().strip()
                            if line == "START":
                                triggered = True
                                break
                        time.sleep(0.1)

                if triggered:
                    with st.spinner(f"Recording {record_seconds}s via Arduino mic..."):
                        audio_path = record_audio_sounddevice(
                            record_seconds, SAMPLE_RATE
                        )
                    with st.spinner("Transcribing with Whisper..."):
                        word_input = transcribe_whisper(audio_path)
                    os.unlink(audio_path)
                    st.success(f"🎤 Heard: **{word_input}**")
                else:
                    st.warning("⏱️ No button press detected — try again")

            else:
                # ── Laptop mic path ───────────────────────────
                with st.spinner(f"Recording {record_seconds}s from laptop mic..."):
                    audio_path = record_audio_sounddevice(record_seconds, SAMPLE_RATE)
                with st.spinner("Transcribing with Whisper..."):
                    word_input = transcribe_whisper(audio_path)
                os.unlink(audio_path)
                st.success(f"🎤 Heard: **{word_input}**")

        # ── Override with typed word if provided ──────────────
        if word_override.strip():
            word_input = word_override.strip().lower()

    with col2:
        st.markdown("**Sign Results**")

        # ── Search & display ──────────────────────────────────
        if word_input:
            video_index = load_video_index()

            if not video_index:
                st.error(f"❌ ISL Dictionary not found at: {VIDEO_DIR}")
            else:
                matches = fuzzy_match_sign(word_input, video_index, top_n=5)

                if not matches:
                    st.warning(f"No matches found for **{word_input}**")
                else:
                    best = matches[0]
                    score = best["score"]
                    word = best["word"]
                    path = best["path"]

                    st.markdown(
                        f"""
                    <div class='result-box'>
                        <div style='font-size:0.7rem; color:#00D4FF;
                                    letter-spacing:2px; text-transform:uppercase;'>
                            Best Match — {score:.0f}% similarity</div>
                        <div class='result-word' style='color:#00D4FF;'>
                            {word.upper()}</div>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

                    # Play best match video
                    if os.path.exists(path):
                        st.video(path)
                        speak_pyttsx3(word)
                    else:
                        st.error(f"Video not found: {path}")

                    # Other matches
                    if len(matches) > 1:
                        st.markdown("**Other close matches:**")
                        for m in matches[1:]:
                            mc1, mc2 = st.columns([3, 1])
                            with mc1:
                                st.markdown(
                                    f"• {m['word'].capitalize()} ({m['score']:.0f}%)"
                                )
                            with mc2:
                                if st.button("▶ Play", key=f"play_{m['word']}"):
                                    if os.path.exists(m["path"]):
                                        st.video(m["path"])
                                        speak_pyttsx3(m["word"])
        else:
            st.markdown(
                """
            <div class='result-box' style='padding:2rem;'>
                <div style='font-size:2rem;'>🔊</div>
                <div style='color:#6b6b8a; font-size:0.9rem; margin-top:0.5rem;'>
                    Press <b>Speak a Sign Word</b> or type a word
                    to find its ISL sign.
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )


# =============================================================
# FOOTER
# =============================================================
st.markdown("---")
st.markdown(
    """
<div style='text-align:center; color:#6b6b8a; font-size:0.75rem;
            letter-spacing:1px; padding-bottom:1rem;'>
    VARTA-SYNC &nbsp;·&nbsp; Bidirectional ISL Interpreter &nbsp;·&nbsp;
    B.Tech Major Project 2026 &nbsp;·&nbsp; BPIT Delhi
</div>
""",
    unsafe_allow_html=True,
)
