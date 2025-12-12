import streamlit as st
from openai import OpenAI
from audio_recorder_streamlit import audio_recorder
import tempfile
import os

st.set_page_config(page_title="Voice Recorder + Whisper Transcribe", layout="centered")

st.title("🎤 Voice Recorder → Save → Whisper Transcribe")
st.write("Click the mic to start recording, click again to stop. Then press 'Save' and 'Transcribe'.")

# --- Load OpenAI key from Streamlit secrets ---
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("OpenAI API key not found. Add OPENAI_API_KEY to Streamlit Secrets and reload.")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

# Session state for file path and raw audio bytes
if "audio_bytes" not in st.session_state:
    st.session_state.audio_bytes = None
if "saved_wav_path" not in st.session_state:
    st.session_state.saved_wav_path = None
if "transcript" not in st.session_state:
    st.session_state.transcript = None

# ---- Recorder widget (Start/Stop) ----
st.subheader("1) Record")
st.write("Press the mic to start recording. Press again to stop.")
audio_bytes = audio_recorder()  # returns bytes (WAV) or None

if audio_bytes is not None:
    st.session_state.audio_bytes = audio_bytes
    st.success("Recording captured — press **Save** to store and then **Transcribe**.")
    st.audio(audio_bytes, format="audio/wav")

# ---- Save recorded audio to a temp WAV ----
st.subheader("2) Save")
if st.button("Save Recording"):
    if not st.session_state.audio_bytes:
        st.warning("No recording found. Please record first.")
    else:
        # Save to a temporary wav file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(st.session_state.audio_bytes)
            tmp_path = tmp.name
        st.session_state.saved_wav_path = tmp_path
        st.success(f"Saved to {tmp_path}")
        st.audio(tmp_path)

# ---- Transcribe with Whisper ----
st.subheader("3) Transcribe")
if st.button("Transcribe"):
    if not st.session_state.saved_wav_path:
        st.warning("No saved recording. Click Save Recording first.")
    else:
        st.info("Sending to Whisper for transcription — please wait...")
        try:
            with open(st.session_state.saved_wav_path, "rb") as f:
                resp = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=f
                )
            # Depending on client return structure; handle both dict-like or object-like
            text = None
            if isinstance(resp, dict):
                text = resp.get("text") or resp.get("data") or str(resp)
            else:
                # object-like (openai client variants)
                text = getattr(resp, "text", None) or (resp.get("text") if hasattr(resp, "get") else None)

            st.session_state.transcript = text or "(no transcript returned)"
            st.success("Transcription complete")
            st.subheader("📝 Transcript")
            st.write(st.session_state.transcript)

        except Exception as e:
            st.error(f"Transcription failed: {e}")

# ---- Optional: Clear saved recording ----
st.subheader("Utilities")
if st.button("Clear Recording + Transcript"):
    # remove temp file if exists
    try:
        if st.session_state.get("saved_wav_path") and os.path.exists(st.session_state["saved_wav_path"]):
            os.remove(st.session_state["saved_wav_path"])
    except Exception:
        pass
    st.session_state.audio_bytes = None
    st.session_state.saved_wav_path = None
    st.session_state.transcript = None
    st.success("Cleared.")
