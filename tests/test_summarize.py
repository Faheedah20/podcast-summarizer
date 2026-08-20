import os
from unittest.mock import patch

from utils.summarize import summarize_transcript, _call_groq


@patch("utils.summarize._call_groq")
def test_summary_is_short_and_focused_on_recording(mock_call):
    mock_call.return_value = '''{
      "title": "Podcast discussion",
      "summary": "This recording discusses the product launch and the team goals for the quarter. The speakers explain the timeline, customer feedback, and the next steps they want to take. They also mention blockers and a few follow-up actions for the team. This section includes background context that is not necessary for the core recording summary. The overall takeaway is that the team is moving forward with clear priorities.",
      "key_points": ["One", "Two", "Three"],
      "important_insights": ["Insight 1", "Insight 2"],
      "discussion_points": [{"timestamp": "00:10:00", "topic": "Launch", "details": "Very detailed discussion"}],
      "action_items": [{"task": "Ship the feature", "owner": "Unassigned", "deadline": null}],
      "speakers": []
    }'''

    with patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}, clear=False):
        result = summarize_transcript("dummy transcript", provider="groq")

    assert result["title"] == "Podcast discussion"
    sentence_count = len([s for s in result["summary"].split(".") if s.strip()])
    assert sentence_count <= 3
    assert "launch" in result["summary"].lower()
    assert "follow-up actions" not in result["summary"].lower()


@patch("utils.summarize.requests.post")
def test_groq_summary_uses_http_request(mock_post):
    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = {
        "choices": [{"message": {"content": '{"title": "Podcast", "summary": "This recording covers the main topic.", "key_points": [], "important_insights": [], "discussion_points": [], "action_items": [], "speakers": []}'}}]
    }

    with patch.dict(os.environ, {"GROQ_API_KEY": "test-key", "GROQ_MODEL": "llama-3.3-70b-versatile"}, clear=False):
        summary = _call_groq("some transcript here")

    assert "This recording covers the main topic." in summary
    assert mock_post.call_count >= 1
