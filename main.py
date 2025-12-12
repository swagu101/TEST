import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from openai import OpenAI
import av
import numpy as np
import tempfile
import wave

st.set_page_config(page_title="Voice to Text", layout="centered")

st.title("🎤 Voice Recorder + Whisper STT")
st.write("Click **Start Recording**, speak, then **Stop**, save the audio, and press **Transcribe**.")

# Load API key
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Session states
if "audio_frames" not in st.session_state:
    st.session_state.audio_frames = []

if "recorded_file" not in st.session_state:
    st.session_state.recorded_file = None


# --------------------------------
# AUDIO FRAME CAPTURE PROCESSOR
# --------------------------------
class RecorderProcessor:
    def recv_audio(self, frame: av.AudioFrame):
        audio = frame.to_ndarray()
        audio = audio.mean(axis=0).astype(np.int16)
        st.session_state.audio_frames.append(audio)
        return frame


# --------------------------------
# START/STOP RECORDER
# --------------------------------
recorder = webrtc_streamer(
    key="recorder",
    mode=WebRtcMode.SENDONLY,
    audio_receiver_size=1024,
    audio_processor_factory=RecorderProcessor,
    media_stream_constraints={"audio": True, "video": False},
)

# --------------------------------
# SAVE AUDIO FUNCTION
# --------------------------------
def save_audio():
    if len(st.session_state.audio_frames) == 0:
        return None

    audio_data = np.concatenate(st.session_state.audio_frames)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        wav_path = tmp.name

        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(44100)
            wf.writeframes(audio_data.tobytes())

    st.session_state.recorded_file = wav_path
    return wav_path


# --------------------------------
# SHOW "SAVE" BUTTON AFTER STOP
# --------------------------------
if recorder and not recorder.state.playing:
    if st.button("Save Recording"):
        file_path = save_audio()
        if file_path:
            st.success("Recording saved successfully!")
            st.audio(file_path)


# --------------------------------
# TRANSCRIBE BUTTON
# --------------------------------
if st.session_state.recorded_file:
    if st.button("Transcribe"):
        st.write("⏳ Transcribing…")

        try:
            result = client.audio.transcriptions.create(
                model="whisper-1",
                file=open(st.session_state.recorded_file, "rb")
            )

            st.success("✔ Transcription Complete!")
            st.subheader("📝 Transcript:")
            st.write(result.text)

        except Exception as e:
            st.error(f"Error: {str(e)}")
