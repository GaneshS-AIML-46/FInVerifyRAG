import pytest
import os
import json
from unittest.mock import patch, MagicMock
from common.llm_client import LLMClient, _role_costs
from common.errors import LLMError
import openai

def _fake_cfg(model="claude-test"):
    """Minimal config for the new role-based LLMClient."""
    return {
        "roles": {
            "generator": {
                "provider": "ollama",
                "model": model,
                "api_key_env": "LLM_API_KEY",
                "max_tokens": 512,
                "temperature": 0.0,
            }
        },
        "llm_prices": {model: {"input": 1.0, "output": 2.0}},
        "budget_caps": {},
    }

def test_llm_client_cache_hit():
    with patch.dict(os.environ, {"LLM_API_KEY": "test-key"}):
        client = LLMClient("generator", _fake_cfg(), check_prices=True)
    with patch.object(client, "_read_cache", return_value={"text": "Hello", "model": "test-model",
                                                            "input_tokens": 5, "output_tokens": 5}):
        res = client.chat(user="Hi")
        assert res["cache_hit"] is True
        assert res["text"] == "Hello"

def test_llm_client_retry():
    model = "claude-test"
    with patch.dict(os.environ, {"LLM_API_KEY": "test-key"}):
        client = LLMClient("generator", _fake_cfg(model), check_prices=True)

    mock_create = MagicMock()
    response_mock = MagicMock()
    response_mock.choices = [MagicMock(message=MagicMock(content="Success"))]
    response_mock.model = model
    response_mock.usage = MagicMock(prompt_tokens=5, completion_tokens=5)

    # 1. Fail with RateLimitError, 2. Fail with 500, 3. Succeed
    mock_create.side_effect = [
        openai.RateLimitError("Rate limited", response=MagicMock(status_code=429), body=None),
        openai.APIStatusError("Server Error", response=MagicMock(status_code=500), body=None),
        response_mock,
    ]
    client._openai_client = MagicMock()
    client._openai_client.chat.completions.create = mock_create

    with patch.object(client, "_read_cache", return_value=None), \
         patch.object(client, "_write_cache"), \
         patch("time.sleep"):
        res = client.chat(user="Hi")
        assert res["text"] == "Success"
        assert mock_create.call_count == 3

def test_resume_logic(tmp_path):
    # Simulate queries.jsonl checkpoint
    checkpoint = tmp_path / "queries.jsonl"
    with open(checkpoint, "w") as f:
        f.write(json.dumps({"question": "Q1", "status": "OK"}) + "\n")
        f.write(json.dumps({"question": "Q2", "status": "ERROR"}) + "\n")
        
    completed = set()
    with open(checkpoint, "r") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                if data.get("status") in ("OK", "ERROR"):
                    completed.add(data.get("question"))
                    
    assert "Q1" in completed
    assert "Q2" in completed
    assert "Q3" not in completed
