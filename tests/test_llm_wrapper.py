import pytest
import os
import json
from unittest.mock import patch, MagicMock
from common.llm_client import LLMClient
import openai

def test_llm_client_cache_hit():
    client = LLMClient()
    with patch.object(client, "_read_cache", return_value={"text": "Hello", "model": "test-model"}):
        res = client.chat(user="Hi")
        assert res["cache_hit"] is True
        assert res["text"] == "Hello"

def test_llm_client_retry():
    client = LLMClient()
    
    mock_create = MagicMock()
    response_mock = MagicMock()
    response_mock.choices = [MagicMock(message=MagicMock(content="Success"))]
    response_mock.model = "test-model"
    response_mock.choices[0].logprobs = None
    
    # 1. Fail with RateLimitError
    # 2. Fail with APIStatusError (500)
    # 3. Succeed
    mock_create.side_effect = [
        openai.RateLimitError("Rate limited", response=MagicMock(status_code=429), body=None),
        openai.APIStatusError("Internal Server Error", response=MagicMock(status_code=500), body=None),
        response_mock
    ]
    
    client.client.chat.completions.create = mock_create
    
    with patch.object(client, "_read_cache", return_value=None), patch.object(client, "_write_cache"):
        # We need to speed up the retry wait time or it will be slow
        with patch('time.sleep', return_value=None):
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
