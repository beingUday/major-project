# ============================================================
# Varta-Sync — Agent 1 Inference (Final Version)
# Sign → Speech
#
# Three-tier confidence routing:
#   > 75%       → speak word directly
#   50% – 75%   → speak "I think you said {word}"
#   ≤ 50%       → speak "Not sure about the exact word,
#                          but here are the top predictions"
#                  → show top-5 with confidence on screen
#
# Works with 75-class ISL model only — no external data dependencies
#
# ⚠️  UPDATE PATHS SECTION BEFORE RUNNING
# ============================================================
import sys
import numpy

if not hasattr(numpy, "_core"):
    sys.modules["numpy._core"] = numpy
    import numpy.core.multiarray as multiarray

    sys.modules["numpy._core.multiarray"] = multiarray


import cv2
import numpy as np
import mediapipe as mp
import pickle
import json
import pyttsx3
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (
    MultiHeadAttention,
    Dense,
    Dropout,
    LayerNormalization,
)
import time
import os
import h5py, shutil, tempfile
import sys

# ============================================================
# ⚠️  UPDATE THESE PATHS
# ============================================================
MODEL_DIR = r"E:/VARTAlaabh/models"
LSTM_MODEL_PATH = f"{MODEL_DIR}/isl_lstm_v2.h5"
TRANS_MODEL_PATH = f"{MODEL_DIR}/isl_transformer_v2.h5"
SCALER_PATH = f"{MODEL_DIR}/scaler.pkl"
LABEL_MAP_PATH = f"{MODEL_DIR}/isl_label_mapping_751.json"
ENSEMBLE_CFG_PATH = f"{MODEL_DIR}/ensemble_config.json"


# Webcam index (0 = default webcam)
WEBCAM_INDEX = 0


# ============================================================
# CONSTANTS — DO NOT CHANGE
# ============================================================
N_FRAMES = 30  # frames per prediction window
N_LANDMARKS = 70  # total landmarks
N_FEATURES = 210  # 70 × 3
HIGH_CONF_THRESHOLD = 0.75  # > 75%  → speak directly
MED_CONF_THRESHOLD = 0.50  # > 50%  → speak with qualifier
COLLECTION_DELAY = 0.033  # ~30fps


# ============================================================
# CUSTOM LAYER DEFINITIONS
# Required to load .keras models
# ============================================================
@tf.keras.utils.register_keras_serializable()
class PositionalEncoding(keras.layers.Layer):
    def __init__(self, max_len, embed_dim, **kwargs):
        super().__init__(**kwargs)
        position = np.arange(max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, embed_dim, 2) * -(np.log(10000.0) / embed_dim))
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
            num_heads=num_heads, key_dim=embed_dim // num_heads, dropout=dropout_rate
        )
        self.ffn = keras.Sequential(
            [Dense(ff_dim, activation="gelu"), Dropout(dropout_rate), Dense(embed_dim)]
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
        x = x + self.drop2(h, training=training)
        return x

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
        ratio = tf.minimum(step / self.warmup_steps, 1.0)
        return self.peak_lr * ratio

    def get_config(self):
        return {"peak_lr": self.peak_lr, "warmup_steps": self.warmup_steps}


CUSTOM = {
    "LinearWarmUp": LinearWarmUp,
    "PositionalEncoding": PositionalEncoding,
    "TransformerBlock": TransformerBlock,
    "AttentionPooling": AttentionPooling,
}


def load_h5_model(path, custom_objects):
    """Patches batch_shape → shape + batch_size before loading."""
    tmp = tempfile.mktemp(suffix=".h5")
    shutil.copy2(path, tmp)

    with h5py.File(tmp, "r+") as f:
        model_config = json.loads(f.attrs["model_config"])

        def fix_layer(obj):
            if isinstance(obj, dict):
                if obj.get("class_name") == "InputLayer":
                    cfg = obj.get("config", {})
                    if "batch_shape" in cfg:
                        batch_shape = cfg.pop("batch_shape")
                        cfg["batch_input_shape"] = batch_shape
                    elif "shape" in cfg:
                        shape = cfg.pop("shape")
                        batch_size = cfg.pop("batch_size", None)
                        cfg["batch_input_shape"] = [batch_size] + list(shape)

                # ← ADD THIS: strip DTypePolicy from any layer
                cfg = obj.get("config", {})
                if "dtype" in cfg and isinstance(cfg["dtype"], dict):
                    if cfg["dtype"].get("class_name") == "DTypePolicy":
                        cfg["dtype"] = cfg["dtype"]["config"].get("name", "float32")

                # === NEW SNIPPET STARTS HERE ===
                # Fix 3: Scrub Keras 3 specific layer arguments
                layer_cfg = obj.get("config", {})
                if isinstance(layer_cfg, dict):
                    # Remove rms_scaling which crashes LayerNormalization in Keras 2
                    if "rms_scaling" in layer_cfg:
                        layer_cfg.pop("rms_scaling")

                    # Scrub 'module' key from initializers (Keras 3 meta-data)
                    for key in [
                        "beta_initializer",
                        "gamma_initializer",
                        "kernel_initializer",
                        "bias_initializer",
                    ]:
                        init_cfg = layer_cfg.get(key)
                        if isinstance(init_cfg, dict) and "module" in init_cfg:
                            init_cfg.pop("module")
                # === NEW SNIPPET ENDS HERE ===

                if "inbound_nodes" in obj:
                    if isinstance(obj["inbound_nodes"], str):
                        obj["inbound_nodes"] = []

                # ← ADD THIS: fix build_config shape issues
                if "build_config" in obj:
                    obj.pop("build_config")

                # ← ADD THIS: fix compile_config
                if "compile_config" in obj:
                    obj.pop("compile_config")
                # ← ADD THIS: convert Keras 3 inbound_nodes to Keras 2 format
                if "inbound_nodes" in obj and isinstance(obj["inbound_nodes"], list):
                    new_nodes = []
                    for node in obj["inbound_nodes"]:
                        if isinstance(node, dict) and "args" in node:
                            # Extract keras_history from each arg
                            for arg in node["args"]:
                                if (
                                    isinstance(arg, dict)
                                    and arg.get("class_name") == "__keras_tensor__"
                                ):
                                    history = arg["config"].get("keras_history", [])
                                    if history:
                                        # Keras 2 format: [layer_name, node_index, tensor_index]
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

        fix_layer(model_config)
        f.attrs["model_config"] = json.dumps(model_config)

    model = keras.models.load_model(tmp, custom_objects=custom_objects, compile=False)
    os.remove(tmp)
    print(f"      ✅ Loaded {os.path.basename(path)}")
    return model


# def load_keras3_model(folder_path, custom_objects):
#     """
#     Load a Keras 3 saved model folder into local Keras 2 environment.
#     Keras 3 saves as config.json + model.weights.h5
#     Keras 2 cannot read this directly — so we rebuild + load weights.
#     """
#     import json

#     config_path = os.path.join(folder_path, "config.json")
#     weights_path = os.path.join(folder_path, "model.weights.h5")

#     if not os.path.exists(config_path):
#         raise FileNotFoundError(f"config.json not found in {folder_path}")
#     if not os.path.exists(weights_path):
#         raise FileNotFoundError(f"model.weights.h5 not found in {folder_path}")

#     with open(config_path) as f:
#         config = json.load(f)

#     model = keras.models.model_from_json(
#         json.dumps(config), custom_objects=custom_objects
#     )
#     model.load_weights(weights_path)
#     print(f"      ✅ Loaded from {os.path.basename(folder_path)}/")
#     return model


# ============================================================
# LANDMARK EXTRACTION
# Order matches training pipeline EXACTLY:
#   lips(20) + left_hand(21) + right_hand(21) + upper_pose(8)
# ============================================================
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
    """
    Extract 70 landmarks in exact training order:
      lips(20) → left_hand(21) → right_hand(21) → upper_pose(8)

    Returns (70, 3) float32 or None if pose not detected.
    """
    # Pose required for shoulder normalization
    if not results.pose_landmarks:
        return None

    pose = results.pose_landmarks.landmark
    lsx = pose[11].x
    lsy = pose[11].y
    rsx = pose[12].x
    rsy = pose[12].y
    cx = (lsx + rsx) / 2.0
    cy = (lsy + rsy) / 2.0
    sd = max(abs(rsx - lsx), 1e-6)

    frame = np.zeros((N_LANDMARKS, 3), dtype=np.float32)

    # ── Lips (0:20) — shoulder-centered ──────────────────────
    if results.face_landmarks:
        face = results.face_landmarks.landmark
        for i, idx in enumerate(LIP_IDX):
            frame[i, 0] = (face[idx].x - cx) / sd
            frame[i, 1] = (face[idx].y - cy) / sd
            frame[i, 2] = face[idx].z

    # ── Left hand bone-relative (20:41) ──────────────────────
    if results.left_hand_landmarks:
        lh = np.array(
            [[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark],
            dtype=np.float32,
        )
        for j, p in HAND_PARENTS:
            frame[20 + j] = lh[j] - lh[p]

    # ── Right hand bone-relative (41:62) ─────────────────────
    if results.right_hand_landmarks:
        rh = np.array(
            [[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark],
            dtype=np.float32,
        )
        for j, p in HAND_PARENTS:
            frame[41 + j] = rh[j] - rh[p]

    # ── Upper pose shoulder-centered (62:70) ──────────────────
    for i, idx in enumerate(POSE_IDX):
        frame[62 + i, 0] = (pose[idx].x - cx) / sd
        frame[62 + i, 1] = (pose[idx].y - cy) / sd
        frame[62 + i, 2] = pose[idx].z

    return frame  # (70, 3)


def resample_frames(frames, target=30):
    """Resample frame list to exactly target frames."""
    n = len(frames)
    if n == target:
        return np.array(frames)
    indices = np.linspace(0, n - 1, target).astype(int)
    return np.array(frames)[indices]


# ============================================================
# AGENT 1 CLASS
# ============================================================
class Agent1:
    def __init__(self):
        print("=" * 50)
        print("  Varta-Sync — Agent 1 (Sign → Speech)")
        print("=" * 50)

        # ── Ensemble config ───────────────────────────────────
        with open(ENSEMBLE_CFG_PATH) as f:
            cfg = json.load(f)
        self.w_lstm = cfg["w_lstm"]
        self.w_trans = cfg["w_trans"]
        print(f"  Weights  — LSTM:{self.w_lstm:.3f}  Trans:{self.w_trans:.3f}")
        print(f"  Accuracy — {cfg['ensemble_acc']:.1%} on val set")

        # ── Load models ───────────────────────────────────────
        # REPLACE WITH THIS
        print("   Loading LSTM...")
        self.lstm = load_h5_model(LSTM_MODEL_PATH, CUSTOM)

        print("   Loading Transformer...")
        self.transformer = load_h5_model(TRANS_MODEL_PATH, CUSTOM)
        # ── Verify output shape ───────────────────────────────
        test = np.random.randn(1, 30, 210).astype("float32")
        lo = self.lstm(test, training=False).numpy()
        to = self.transformer(test, training=False).numpy()
        assert lo.shape == (1, 75), (
            f"❌ LSTM wrong shape {lo.shape} — are you loading ISL model?"
        )
        assert to.shape == (1, 75), (
            f"❌ Transformer wrong shape {to.shape} — are you loading ISL model?"
        )
        print(f"  Models verified — output: (1, 75) ✅")

        # ── Scaler ────────────────────────────────────────────
        with open(SCALER_PATH, "rb") as f:
            self.scaler = pickle.load(f)
        print("  Scaler loaded ✅")

        # ── Label mapping ─────────────────────────────────────
        with open(LABEL_MAP_PATH) as f:
            sign_to_idx = json.load(f)
        self.idx_to_sign = {int(v): k for k, v in sign_to_idx.items()}
        print(f"  Labels — {len(self.idx_to_sign)} classes ✅")

        # ── TTS ───────────────────────────────────────────────
        self.tts = pyttsx3.init()
        self.tts.setProperty("rate", 150)
        print("  TTS ready ✅")

        # ── MediaPipe ─────────────────────────────────────────
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils

        print("=" * 50)
        print("  ✅ Agent 1 ready")
        print("=" * 50 + "\n")

    # ─────────────────────────────────────────────────────────
    # PREPROCESSING
    # ─────────────────────────────────────────────────────────
    def preprocess(self, frames_raw):
        """
        frames_raw : list of (70, 3) — raw MediaPipe output
        Returns    : (1, 30, 210) scaled float32
        """
        frames = resample_frames(frames_raw, target=N_FRAMES)
        x = frames.reshape(1, N_FRAMES, N_FEATURES)
        x_scaled = (
            self.scaler.transform(x.reshape(1, -1))
            .reshape(1, N_FRAMES, N_FEATURES)
            .astype(np.float32)
        )
        return x_scaled

    # ─────────────────────────────────────────────────────────
    # ENSEMBLE PREDICTION
    # ─────────────────────────────────────────────────────────
    def predict(self, x_scaled):
        """
        Returns top-5 list with sign and confidence.
        """
        lstm_p = self.lstm(x_scaled, training=False).numpy()[0]  # (75,)
        trans_p = self.transformer(x_scaled, training=False).numpy()[0]  # (75,)
        combined = self.w_lstm * lstm_p + self.w_trans * trans_p

        top5_idx = np.argsort(combined)[::-1][:5]
        top5 = [
            {
                "sign": self.idx_to_sign[int(i)],
                "confidence": float(combined[i]),
                "index": int(i),
            }
            for i in top5_idx
        ]

        return top5

    # ─────────────────────────────────────────────────────────
    # TTS
    # ─────────────────────────────────────────────────────────
    def speak(self, text):
        """Speak text using pyttsx3."""
        print(f"🔊 {text}")
        self.tts.say(text)
        self.tts.runAndWait()

    # ─────────────────────────────────────────────────────────
    # THREE-TIER CONFIDENCE ROUTER
    # ─────────────────────────────────────────────────────────
    def route(self, x_scaled):
        """
        Tier 1: confidence > 75%
                → speak word directly

        Tier 2: confidence 50–75%
                → speak "I think you said {word}"
                → show top-5 on screen

        Tier 3: confidence ≤ 50%
                → speak "Not sure about the exact word,
                          but here are the top predictions"
                → show top-5 on screen
        """
        top5 = self.predict(x_scaled)
        top1 = top5[0]
        top1_conf = top1["confidence"]
        top1_sign = top1["sign"]

        # Console log
        print(f"\n{'=' * 50}")
        print(f"  Confidence : {top1_conf:.1%}")
        print(f"  Top-1      : {top1_sign}")
        print(f"  Top-5      :")
        for i, p in enumerate(top5):
            bar = "█" * int(p["confidence"] * 20)
            print(f"    #{i + 1} {p['sign']:25s} {p['confidence']:.1%} {bar}")
        print(f"{'=' * 50}")

        # ── Tier 1: High confidence ───────────────────────────
        if top1_conf > HIGH_CONF_THRESHOLD:
            print(f"  ✅ Tier 1 — speaking directly")
            self.speak(top1_sign)
            return {
                "action": "spoke",
                "tier": 1,
                "sign": top1_sign,
                "spoken": top1_sign,
                "confidence": top1_conf,
                "top5": top5,
            }

        # ── Tier 2: Medium confidence ─────────────────────────
        elif top1_conf > MED_CONF_THRESHOLD:
            spoken = f"I think you said {top1_sign}"
            print(f"  🟡 Tier 2 — speaking with qualifier")
            self.speak(spoken)
            return {
                "action": "uncertain",
                "tier": 2,
                "sign": top1_sign,
                "spoken": spoken,
                "confidence": top1_conf,
                "top5": top5,
            }

        # ── Tier 3: Low confidence ────────────────────────────
        else:
            spoken = "Not sure about the exact word, but here are the top predictions"
            print(f"  ❓ Tier 3 — low confidence, showing top-5")
            self.speak(spoken)
            return {
                "action": "show_options",
                "tier": 3,
                "sign": None,
                "spoken": spoken,
                "confidence": top1_conf,
                "top5": top5,
            }

    # ─────────────────────────────────────────────────────────
    # STANDALONE WEBCAM LOOP
    # ─────────────────────────────────────────────────────────
    def run(self):
        """
        Standalone real-time loop — runs without Streamlit.
        SPACE → collect 30 frames → predict → route
        Q     → quit
        """
        cap = cv2.VideoCapture(WEBCAM_INDEX)
        if not cap.isOpened():
            print("❌ Cannot open webcam — check WEBCAM_INDEX")
            return

        frame_buffer = []
        collecting = False
        status_text = "Press SPACE to start signing"
        last_result = None

        with self.mp_holistic.Holistic(
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        ) as holistic:
            print("🎥 Webcam open")
            print("   SPACE → start collecting")
            print("   Q     → quit\n")

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(rgb)

                # Draw landmarks
                self.mp_drawing.draw_landmarks(
                    frame,
                    results.left_hand_landmarks,
                    mp.solutions.holistic.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(
                        color=(121, 22, 76), thickness=2, circle_radius=2
                    ),
                    self.mp_drawing.DrawingSpec(
                        color=(121, 44, 250), thickness=2, circle_radius=1
                    ),
                )
                self.mp_drawing.draw_landmarks(
                    frame,
                    results.right_hand_landmarks,
                    mp.solutions.holistic.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(
                        color=(245, 117, 66), thickness=2, circle_radius=2
                    ),
                    self.mp_drawing.DrawingSpec(
                        color=(245, 66, 230), thickness=2, circle_radius=1
                    ),
                )

                # ── Collecting ────────────────────────────────
                if collecting:
                    lm = extract_landmarks(results)
                    if lm is not None:
                        frame_buffer.append(lm)

                    n_col = len(frame_buffer)
                    progress = int((n_col / N_FRAMES) * frame.shape[1])
                    cv2.rectangle(
                        frame,
                        (0, frame.shape[0] - 20),
                        (progress, frame.shape[0]),
                        (0, 255, 100),
                        -1,
                    )
                    status_text = f"Collecting: {n_col}/{N_FRAMES}"

                    if n_col >= N_FRAMES:
                        collecting = False
                        status_text = "Processing..."
                        cv2.putText(
                            frame,
                            status_text,
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2,
                        )
                        cv2.imshow("Agent 1", frame)
                        cv2.waitKey(1)

                        x_scaled = self.preprocess(frame_buffer)
                        last_result = self.route(x_scaled)
                        frame_buffer = []
                        status_text = "Press SPACE to sign again"

                # ── Show last result overlay ──────────────────
                if last_result:
                    action = last_result["action"]
                    sign = last_result.get("sign") or "Unknown"
                    conf = last_result["confidence"]

                    if action == "spoke":
                        color = (0, 255, 100)
                        label = f"{sign.upper()} ({conf:.0%})"
                    elif action == "uncertain":
                        color = (0, 200, 255)
                        label = f"~{sign.upper()} ({conf:.0%})"
                    else:
                        color = (0, 100, 255)
                        label = f"Low conf ({conf:.0%}) — see top-5"

                    cv2.putText(
                        frame,
                        label,
                        (10, frame.shape[0] - 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        color,
                        2,
                    )

                # ── Status + controls overlay ─────────────────
                cv2.putText(
                    frame,
                    status_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0) if collecting else (255, 255, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    "SPACE:collect  Q:quit",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (180, 180, 180),
                    1,
                )

                cv2.imshow("Agent 1 — Sign to Speech", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord(" "):
                    if not collecting:
                        frame_buffer = []
                        collecting = True
                        last_result = None
                        print("🎬 Collecting frames...")
                    else:
                        collecting = False
                        frame_buffer = []
                        status_text = "Cancelled — SPACE to retry"

                time.sleep(COLLECTION_DELAY)

        cap.release()
        cv2.destroyAllWindows()
        print("👋 Agent 1 stopped")

    # ─────────────────────────────────────────────────────────
    # FOR STREAMLIT (app.py calls this)
    # ─────────────────────────────────────────────────────────
    def predict_from_frames(self, frames_raw):
        """
        Called by app.py instead of run().
        frames_raw : list of (70, 3) arrays from MediaPipe
        Returns    : result dict from route()
        """
        if len(frames_raw) < 15:
            return {
                "action": "insufficient_frames",
                "tier": 0,
                "sign": None,
                "spoken": None,
                "confidence": 0.0,
                "top5": [],
                "message": (
                    f"Only {len(frames_raw)} frames collected. "
                    f"Sign more slowly and hold the sign."
                ),
            }
        x_scaled = self.preprocess(frames_raw)
        return self.route(x_scaled)


with h5py.File(r"E:/VARTAlaabh/models/isl_lstm_v2.h5", "r") as f:
    config = json.loads(f.attrs["model_config"])


def find_inbound_nodes(obj, path=""):
    if isinstance(obj, dict):
        if "inbound_nodes" in obj:
            print(f"\nPath: {path}")
            print(f"inbound_nodes type: {type(obj['inbound_nodes'])}")
            print(f"inbound_nodes value: {json.dumps(obj['inbound_nodes'])[:300]}")
        for k, v in obj.items():
            find_inbound_nodes(v, path + f".{k}")
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            find_inbound_nodes(item, path + f"[{i}]")


# find_inbound_nodes(config)

# sys.exit(0)  # ← stops here, won't try to load models


# ============================================================
# STANDALONE RUN
# ============================================================
if __name__ == "__main__":
    agent = Agent1()
    agent.run()
