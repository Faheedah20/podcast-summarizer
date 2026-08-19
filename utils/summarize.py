"""
LLM summarization + action item extraction.
Supports Groq (recommended free/fast) and OpenAI.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any

from utils.config import load_project_env

load_project_env(Path(__file__).resolve().parent.parent)


SYSTEM_PROMPT = """You are an expert meeting and podcast analyst.
Your job is to turn a raw transcript into clear, actionable output.

Always respond with valid JSON in exactly this structure:
{
  "title": "A short descriptive title for the meeting/podcast",
  "summary": "A concise 3-6 sentence executive summary",
  "key_points": [
    "Key point 1",
    "Key point 2",
    "..."
  ],
    "important_insights": [
        "Insight grounded in the transcript, with a timestamp when available"
    ],
    "discussion_points": [
        {
            "timestamp": "HH:MM:SS",
            "topic": "Main topic discussed",
            "details": "Important details from this part of the conversation"
        }
    ],
  "action_items": [
    {
      "task": "What needs to be done",
      "owner": "Person responsible (or 'Unassigned' if not mentioned)",
      "deadline": "Deadline if mentioned, otherwise null"
    }
  ],
  "speakers": ["Speaker names if identifiable, otherwise empty list"]
}

Rules:
- Be accurate. Do not invent information that is not in the transcript.
- Action items should be specific and actionable.
- Important insights should explain implications, conclusions, or notable takeaways.
- Discussion points must cover the main topics and include the relevant transcript timestamp.
- Use the timestamp shown at the start of transcript lines; do not invent precision.
- If no clear action items exist, return an empty list.
- If no important insights or discussion points exist, return an empty list.
- Keep the summary neutral and professional.
"""


def _call_groq(transcript: str) -> str:
    from groq import Groq

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    request = {
        "model": os.getenv("GROQ_MODEL", "qwen/qwen3.6-27b"),
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Transcript:\n\n{transcript}"}
        ],
        "temperature": 0.2,
    }

    # This model can reject forced JSON mode before generating a response.
    response = client.chat.completions.create(**request)

    return response.choices[0].message.content


def _call_openai(transcript: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Transcript:\n\n{transcript}"}
        ],
        temperature=0.2,
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
            "title": "Summary",
            "summary": raw,
            "key_points": [],
            "important_insights": [],
            "discussion_points": [],
            "action_items": [],
            "speakers": []
        }

    data.setdefault("title", "Untitled")
    data.setdefault("summary", "")
    data.setdefault("key_points", [])
    data.setdefault("important_insights", [])
    data.setdefault("discussion_points", [])
    data.setdefault("action_items", [])
    data.setdefault("speakers", [])

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