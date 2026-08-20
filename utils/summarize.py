"""
LLM summarization + action item extraction.
Supports Groq (recommended free/fast) and OpenAI.
"""

import os
import json
import re
from pathlib import Path
from typing import Dict, Any

import requests

from utils.config import load_project_env

load_project_env(Path(__file__).resolve().parent.parent)


SYSTEM_PROMPT = """You are summarizing an uploaded recording.
Focus only on what is clearly in the recording itself.
Return valid JSON with this exact shape:
{
  "title": "Short title based on the recording",
  "summary": "2 short sentences max explaining the main topic and takeaway from the recording",
  "key_points": [],
  "important_insights": [],
  "discussion_points": [],
  "action_items": [],
  "speakers": []
}

Rules:
- Keep the summary brief: 2 sentences maximum.
- Focus on the recording's actual content, not generic meeting boilerplate.
- Do not invent details, speakers, or deadlines.
- If an item is not clearly present, leave it empty.
- Avoid action items, long notes, and unnecessary business-analysis language.
- The summary should sound like a quick description of the uploaded recording.
"""

MAX_SUMMARY_SENTENCES = int(os.getenv("MAX_SUMMARY_SENTENCES", "2"))
MAX_TRANSCRIPT_CHARS = int(os.getenv("MAX_TRANSCRIPT_CHARS", "3000"))


def _prepare_transcript_for_summary(transcript: str) -> str:
    """Keep the transcript short enough for a fast, focused summary."""
    text = (transcript or "").strip()
    if len(text) <= MAX_TRANSCRIPT_CHARS:
        return text

    truncated = text[:MAX_TRANSCRIPT_CHARS].rsplit("\n", 1)[0].strip()
    if not truncated:
        return text[:MAX_TRANSCRIPT_CHARS]
    return truncated + "\n[Transcript shortened for a quick summary.]"


def _shorten_summary(summary: str) -> str:
    """Trim long model output down to a recording-focused summary."""
    cleaned = re.sub(r"\s+", " ", (summary or "")).strip()
    if not cleaned:
        return "This recording covers the main topic discussed in the audio."

    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", cleaned) if part.strip()]
    selected = []
    for sentence in sentences:
        lowered = sentence.lower()
        if any(token in lowered for token in ["overall takeaway", "this section includes", "follow-up actions", "action items", "important insights"]):
            continue
        selected.append(sentence)
        if len(selected) >= MAX_SUMMARY_SENTENCES:
            break

    if not selected:
        return cleaned[:220].rstrip(". ") + "."

    return " ".join(selected)


def _call_groq(transcript: str) -> str:
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key or api_key.startswith("your_"):
        raise ValueError("GROQ_API_KEY is missing or still set to the placeholder in .env")

    candidates = [
        os.getenv("GROQ_MODEL"),
        "qwen/qwen3.6-27b",
        "groq/compound",
        "openai/gpt-oss-20b",
        "openai/gpt-oss-120b",
    ]
    seen = set()

    for model in candidates:
        if not model or model in seen:
            continue
        seen.add(model)

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Transcript:\n\n{_prepare_transcript_for_summary(transcript)}"}
            ],
            "temperature": 0.1,
            "max_tokens": 80,
        }

        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=60,
            )
            if response.status_code >= 400:
                message = response.text
                if "model_not_found" not in message.lower() and "404" not in message:
                    response.raise_for_status()
                continue
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except requests.RequestException:
            continue

    raise RuntimeError("Groq summarization failed: no valid Groq model was available for this request.")


def _call_openai(transcript: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Transcript:\n\n{_prepare_transcript_for_summary(transcript)}"}
        ],
        temperature=0.1,
        max_tokens=80,
        response_format={"type": "json_object"}
    )
    return response.choices[0].message.content


def summarize_transcript(transcript: str, provider: str = None) -> Dict[str, Any]:
    """
    Generate structured summary + action items from transcript.
    provider: "groq" | "openai"
    """
    provider = (provider or os.getenv("LLM_PROVIDER", "groq")).lower()

    if provider == "openai":
        if not _is_configured_key("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY is required")
        raw = _call_openai(transcript)
    else:
        if not _is_configured_key("GROQ_API_KEY"):
            raise ValueError("GROQ_API_KEY is missing or still set to the placeholder in .env")
        raw = _call_groq(transcript)

    try:
        data = _parse_json_response(raw)
    except json.JSONDecodeError:
        data = {
            "title": "Recording summary",
            "summary": raw,
            "key_points": [],
            "important_insights": [],
            "discussion_points": [],
            "action_items": [],
            "speakers": []
        }

    data.setdefault("title", "Recording summary")
    data["summary"] = _shorten_summary(str(data.get("summary", "")))
    data["key_points"] = list(data.get("key_points", []) or [])[:3]
    data["important_insights"] = list(data.get("important_insights", []) or [])[:2]
    data["discussion_points"] = list(data.get("discussion_points", []) or [])[:2]
    data["action_items"] = list(data.get("action_items", []) or [])[:3]
    data["speakers"] = list(data.get("speakers", []) or [])[:5]
    return data


def _parse_json_response(raw: str) -> Dict[str, Any]:
    """Parse JSON returned either directly or inside a Markdown code fence."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned
        cleaned = cleaned.rsplit("```", 1)[0].strip()
    return json.loads(cleaned)


def _is_configured_key(name: str) -> bool:
    """Return false for missing or copied example credentials."""
    value = os.getenv(name, "").strip()
    return bool(value) and not value.startswith("your_")