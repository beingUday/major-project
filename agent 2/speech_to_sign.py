# ============================================================
# VARTA-SYNC — Agent 2: Speech → Sign
# ============================================================
# Pipeline:
#   Microphone → Whisper STT → rapidfuzz match → OpenCV playback
#
# Folder structure expected:
#   islrtc_videos/
#       A_bird_in_the_hand_is_worth_two_in_the_bush.mp4
#       A_blessing_in_disguise.mp4
#       Namaste.mp4
#       Water.mp4
#       ...  (all videos in one flat folder)
#
# Usage:
#   python agent2_speech_to_sign.py
#
# Requirements:
#   pip install -r requirements.txt
# ============================================================

import os
import time
import tempfile
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import cv2
import whisper
import sounddevice as sd
import soundfile as sf
from rapidfuzz import process, fuzz

# ============================================================
# CONFIG — edit these paths for your setup
# ============================================================

VIDEO_DIR = "E:\VARTAlaabh\ISL DATA"  # folder containing all ISLRTC .mp4 files
WHISPER_MODEL = "base"  # tiny / base / small — base is best balance
# on CPU. Use "small" if you have GPU.
RECORD_SECONDS = 4  # how many seconds to record mic input
SAMPLE_RATE = 16000  # Whisper expects 16kHz
FUZZY_THRESHOLD = 70  # minimum match score (0-100)
# lower = more lenient, higher = stricter

# ============================================================
# STEP 1 — BUILD VIDEO INDEX
# Index all .mp4 filenames in VIDEO_DIR at startup.
# Converts "A_Line_Dress.mp4" → "a line dress" for matching.
# ============================================================


def build_video_index(video_dir):
    """
    Walk VIDEO_DIR recursively through alphabet subfolders and build:
        index = { "a line dress": "/path/to/A/A_Line_Dress.mp4", ... }

    Expected structure:
        ISL Dictionary/
            A/
                A_Line_Dress.mp4
                A_Lot.mp4
            B/
                Baby.mp4
            ...
    """
    if not os.path.exists(video_dir):
        raise FileNotFoundError(
            f"Video folder not found: {video_dir}\n"
            f"Set VIDEO_DIR to the 'ISL Dictionary' folder."
        )

    index = {}

    # os.walk recursively visits every subfolder
    for root, dirs, files in os.walk(video_dir):
        for fname in files:
            if not fname.lower().endswith(".mp4"):
                continue
            # Strip extension, replace underscores with spaces, lowercase
            clean_name = fname.rsplit(".", 1)[0]  # remove .mp4
            clean_name = clean_name.replace("_", " ")  # underscores → spaces
            clean_name = clean_name.lower().strip()  # lowercase
            full_path = os.path.join(root, fname)
            index[clean_name] = full_path

    print(f"✅ Video index built: {len(index):,} signs loaded")
    print(f"   Subfolders scanned: {video_dir}")
    return index


# ============================================================
# STEP 2 — MICROPHONE RECORDING
# Records RECORD_SECONDS of audio from default mic.
# Saves to a temp .wav file for Whisper to read.
# ============================================================


def record_audio(duration=RECORD_SECONDS, sample_rate=SAMPLE_RATE):
    """
    Record from default microphone.
    Returns path to saved .wav temp file.
    """
    print(f"\n🎤 Recording for {duration} seconds... Speak now!")

    # Countdown so user knows when to speak
    for i in range(3, 0, -1):
        print(f"   {i}...")
        time.sleep(1)
    print("   🔴 GO!")

    audio = sd.rec(
        int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="float32"
    )
    sd.wait()  # block until recording is done
    print("   ⏹️  Recording stopped.")

    # Save to temp file
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, audio, sample_rate)
    return tmp.name


# ============================================================
# STEP 3 — WHISPER STT
# Transcribes the recorded audio file.
# Returns cleaned text e.g. "namaste" or "a line dress"
# ============================================================


def transcribe_audio(audio_path, model):
    """
    Run Whisper on audio_path.
    Returns lowercase stripped transcription string.
    """
    print("🧠 Transcribing with Whisper...")
    result = model.transcribe(
        audio_path,
        language="en",  # force English — ISL dictionary uses English words
        fp16=False,  # fp16=False for CPU; set True if NVIDIA GPU available
    )
    text = result["text"].strip().lower()

    # Clean up punctuation whisper sometimes adds
    text = text.replace(".", "").replace(",", "").replace("?", "").replace("!", "")
    text = text.strip()

    print(f'📝 Whisper heard: "{text}"')
    return text


# ============================================================
# STEP 4 — FUZZY MATCHING
# Matches spoken text against video index keys.
# Returns (best_match_key, score, video_path) or None.
# ============================================================


def fuzzy_match(spoken_text, video_index, threshold=FUZZY_THRESHOLD):
    """
    Find the best matching sign video for spoken_text.

    Uses token_sort_ratio so word order doesn't matter:
        "dress line a" still matches "a line dress"

    Returns:
        (matched_label, score, video_path)  if score >= threshold
        None                                 if no match found
    """
    if not spoken_text:
        return None

    candidates = list(video_index.keys())

    # Primary match — token sort handles word order variation
    result = process.extractOne(spoken_text, candidates, scorer=fuzz.token_sort_ratio)

    if result is None:
        return None

    matched_label, score, _ = result

    if score >= threshold:
        video_path = video_index[matched_label]
        print(f'✅ Matched: "{matched_label}"  (score: {score}/100)')
        print(f"   📁 Video: {os.path.basename(video_path)}")
        return matched_label, score, video_path
    else:
        print(f'❌ No confident match found for "{spoken_text}"')
        print(f'   Best guess was "{matched_label}" with score {score}/100')
        print(f"   (threshold is {threshold} — try speaking more clearly)")
        return None


# ============================================================
# STEP 5 — VIDEO PLAYBACK
# Plays the matched .mp4 file using OpenCV.
# Press 'q' to quit playback early.
# ============================================================


def play_video(video_path, label):
    """
    Play an MP4 video file in an OpenCV window.
    Press 'q' to quit early.
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25  # fallback if metadata is missing

    delay = int(1000 / fps)  # ms per frame
    window_name = f"Varta-Sync Agent 2 — {label.title()}"

    print(f"\n▶️  Playing: {label.title()}")
    print("   Press 'q' to skip to next sign | Close window to stop")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # video ended

        cv2.imshow(window_name, frame)

        # q = quit early
        if cv2.waitKey(delay) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("   ✅ Playback complete")


# ============================================================
# STEP 6 — FALLBACK: show top 3 suggestions
# When fuzzy match score is below threshold, show the closest
# matches so the user can pick manually.
# ============================================================


def show_suggestions(spoken_text, video_index, top_n=3):
    """
    Show top N closest matches with their scores.
    User can pick one by number or skip.
    """
    candidates = list(video_index.keys())
    results = process.extract(
        spoken_text, candidates, scorer=fuzz.token_sort_ratio, limit=top_n
    )

    if not results:
        print("   No suggestions available.")
        return None

    print(f"\n💡 Did you mean one of these?")
    for i, (label, score, _) in enumerate(results, 1):
        print(f"   {i}. {label.title()}  (score: {score})")

    print(f"   0. None of these — skip")

    while True:
        try:
            choice = int(input("\n   Enter number: ").strip())
            if choice == 0:
                return None
            if 1 <= choice <= len(results):
                chosen_label = results[choice - 1][0]
                return chosen_label, results[choice - 1][1], video_index[chosen_label]
        except (ValueError, KeyboardInterrupt):
            return None


# ============================================================
# MAIN LOOP
# ============================================================


def main():
    print("=" * 60)
    print("  VARTA-SYNC — Agent 2: Speech → Indian Sign Language")
    print("=" * 60)

    # ── Load Whisper model (downloads ~75MB for base, once) ──
    print(f"\n⏳ Loading Whisper '{WHISPER_MODEL}' model...")
    model = whisper.load_model(WHISPER_MODEL)
    print(f"✅ Whisper loaded")

    # ── Build video index ────────────────────────────────────
    video_index = build_video_index(VIDEO_DIR)

    print(f"\n{'─' * 60}")
    print(f"  Ready! Speak a word and Agent 2 will show its ISL sign.")
    print(f"  Type 'q' and press Enter at any time to quit.")
    print(f"{'─' * 60}")

    # ── Main loop ─────────────────────────────────────────────
    while True:
        try:
            # --- NEW: Choice Selection ---
            print("\n[1] Speak word (Voice) | [2] Type word (Text)")
            choice = input("Choice (1/2) or 'q' to quit: ").strip().lower()

            if choice == "q":
                break

            spoken_text = ""  # Initialize empty string

            # --- CHANGE 1: Logic Branching ---
            if choice == "1":
                # Original voice logic
                audio_path = record_audio()
                spoken_text = transcribe_audio(audio_path, model)
                try:
                    os.unlink(audio_path)
                except:
                    pass

            elif choice == "2":
                # New text input logic
                spoken_text = input("👉 Enter word/phrase: ").strip().lower()

            else:
                print("⚠️ Please enter 1 or 2.")
                continue

            # --- CHANGE 2: Unified Input Handling ---
            if not spoken_text:
                print("⚠️ No input detected.")
                continue

            # Step 3: Fuzzy match
            match = fuzzy_match(spoken_text, video_index)

            # Step 4: If no confident match, show suggestions
            if match is None:
                match = show_suggestions(spoken_text, video_index)

            # Step 5: Play video
            if match is not None:
                label, score, video_path = match
                play_video(video_path, label)
            else:
                print(f'\n⚠️  No sign found for "{spoken_text}".')
                print("    Try rephrasing or speaking more clearly.")

        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Exiting Varta-Sync Agent 2.")
            break


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
