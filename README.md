## Podcast Summarizer

### Run locally

```powershell
cd C:\Users\pelum\Desktop\Podcast-summarizer
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open `http://localhost:8501` in your browser.

### How to use

1. Choose **Upload recording** and select an audio file, or choose **Record from microphone**.
2. Allow microphone access when prompted.
3. Select a transcription method and LLM provider in the sidebar.
4. Click **Generate Summary**.

The transcript is formatted with `[HH:MM:SS - HH:MM:SS]` ranges. The generated result includes the executive summary, key points, important insights, timestamped key discussion points, speakers, and action items.

### API keys

Copy `.env.example` to `.env` and replace `your_groq_api_key_here` with your Groq key. A Groq key is enough for both transcription and summarization; an OpenAI key is only required when selecting OpenAI in the sidebar. The default Groq summarization model is `qwen/qwen3.6-27b`. Restart Streamlit after changing `.env`.

Local transcription uses the English-specific `tiny.en` Whisper model, voice-activity detection, and single-pass decoding for faster CPU processing. The first local run downloads the model. For the fastest transcription, add `GROQ_API_KEY` or `OPENAI_API_KEY` to `.env` and select the matching provider; Groq is selected automatically when its key is available.

