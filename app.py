"""
AI Meeting / Podcast Summarizer + Action Items
Streamlit frontend
"""

import os
import streamlit as st
from pathlib import Path

from utils.config import load_project_env

BASE_DIR = Path(__file__).resolve().parent
load_project_env(BASE_DIR)

try:
    for name, value in st.secrets.items():
        if value is not None:
            os.environ[name] = str(value)
except Exception:
    pass

from utils.transcribe import transcribe_audio, save_uploaded_file
from utils.summarize import summarize_transcript

groq_key = os.getenv("GROQ_API_KEY", "").strip()
groq_configured = bool(groq_key) and not groq_key.startswith("your_")


def _format_readable_summary(result):
    """Create a plain-text download without exposing the internal JSON structure."""
    lines = [result.get("title", "Summary"), "", "SUMMARY", result.get("summary", "")]

    sections = [
        ("KEY POINTS", result.get("key_points", [])),
        ("IMPORTANT INSIGHTS", result.get("important_insights", [])),
    ]
    for heading, items in sections:
        if items:
            lines.extend(["", heading, *[f"- {item}" for item in items]])

    discussion_points = result.get("discussion_points", [])
    if discussion_points:
        lines.extend(["", "KEY DISCUSSION POINTS"])
        for point in discussion_points:
            lines.append(
                f"- [{point.get('timestamp', 'Timestamp unavailable')}] "
                f"{point.get('topic', 'Untitled discussion')}: {point.get('details', '')}"
            )

    action_items = result.get("action_items", [])
    if action_items:
        lines.extend(["", "ACTION ITEMS"])
        for item in action_items:
            lines.append(
                f"- {item.get('task', '')} | Owner: {item.get('owner', 'Unassigned')}"
            )

    return "\n".join(lines).strip()

st.set_page_config(
    page_title="AI Meeting & Podcast Summarizer",
    page_icon="🎙️",
    layout="wide"
)

# Sidebar
st.sidebar.title("⚙️ Settings")
if not groq_configured:
    st.sidebar.error(
        "Groq is not configured. Put your real GROQ_API_KEY in the project's .env file."
    )
transcription_method = st.sidebar.selectbox(
    "Transcription method",
    options=["local", "groq", "openai"],
    index=1 if os.getenv("GROQ_API_KEY") else 0,
    help="Groq is fastest when GROQ_API_KEY is configured. Local uses a speed-optimized Whisper model."
)

llm_provider = st.sidebar.selectbox(
    "LLM provider",
    options=["groq", "openai"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **How to use**
    1. Choose **Upload recording** or **Record from microphone**.
    2. Allow microphone access if recording in the browser.
    3. Click **Generate Summary**.
    4. Review or download the results.
    """
)

# Main UI
st.title("🎙️ AI Meeting & Podcast Summarizer")
st.markdown(
    "Upload a meeting recording or podcast episode and get a clean summary + actionable next steps."
)

upload_tab, record_tab = st.tabs(["Upload recording", "Record from microphone"])

with upload_tab:
    uploaded_file = st.file_uploader(
        "Upload a recording",
        type=[
            "mp3", "wav", "m4a", "aac", "ogg", "flac", "webm", "mp4", "mpeg", "mpga"
        ],
        help="Choose an audio or video recording. Files up to 500 MB are supported."
    )

with record_tab:
    recorded_file = st.audio_input("Record audio")

selected_file = recorded_file or uploaded_file

if selected_file is not None:
    st.audio(selected_file)

    if st.button("🚀 Generate Summary", type="primary", use_container_width=True):
        with st.spinner("Transcribing audio… this can take a minute"):
            try:
                audio_path = save_uploaded_file(selected_file)
                transcript = transcribe_audio(audio_path, method=transcription_method)
                try:
                    os.unlink(audio_path)
                except Exception:
                    pass
            except Exception as e:
                st.error(f"Transcription failed ({transcription_method}): {type(e).__name__}: {e}")
                st.stop()

        with st.spinner("Generating summary and extracting action items…"):
            try:
                result = summarize_transcript(transcript, provider=llm_provider)
            except Exception as e:
                st.error(f"Summarization failed ({llm_provider}): {type(e).__name__}: {e}")
                st.stop()

        st.success("Done!")
        st.header(result.get("title", "Summary"))

        st.subheader("📄 Summary")
        st.write(result.get("summary", ""))

        important_insights = result.get("important_insights", [])
        if important_insights:
            st.subheader("💡 Important Insights")
            for insight in important_insights:
                st.markdown(f"- {insight}")

        key_points = result.get("key_points", [])
        if key_points:
            st.subheader("🔑 Key Points")
            for point in key_points:
                st.markdown(f"- {point}")

        discussion_points = result.get("discussion_points", [])
        if discussion_points:
            st.subheader("🗣️ Key Discussion Points")
            for point in discussion_points:
                timestamp = point.get("timestamp", "Timestamp unavailable")
                topic = point.get("topic", "Untitled discussion")
                details = point.get("details", "")
                st.markdown(f"**[{timestamp}] {topic}**  \n{details}")

        action_items = result.get("action_items", [])
        st.subheader("✅ Action Items")
        if action_items:
            for i, item in enumerate(action_items, 1):
                task = item.get("task", "")
                owner = item.get("owner", "Unassigned")
                deadline = item.get("deadline")
                deadline_str = f" | Deadline: {deadline}" if deadline else ""
                st.markdown(f"**{i}. {task}**  \nOwner: {owner}{deadline_str}")
        else:
            st.info("No clear action items found.")

        speakers = result.get("speakers", [])
        if speakers:
            st.subheader("🗣️ Speakers")
            st.write(", ".join(speakers))

        st.subheader("🕒 Transcript with timestamps")
        st.text_area("Transcript", transcript, height=350, label_visibility="collapsed")

        readable_summary = _format_readable_summary(result)

        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "Download Summary",
                data=readable_summary,
                file_name="summary.txt",
                mime="text/plain"
            )
        with col2:
            st.download_button(
                "Download Transcript",
                data=transcript,
                file_name="transcript.txt",
                mime="text/plain"
            )
else:
    st.info("👆 Upload an audio file to get started.")

st.markdown("---")