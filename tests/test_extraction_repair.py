import pytest
import json
from unittest.mock import MagicMock
from atoms.llm_extract import LLMAtomExtractor
from common.schemas import GeneratedAnswer

def test_json_parse_repair():
    llm_mock = MagicMock()
    # First response: prose with malformed json
    # Second response: valid json
    
    first_response = "Here are the atoms you requested:\\n```json\\n[{\"type\": \"RATE\", \"text\": \"5%\"} {invalid_json...]\\n```"
    second_response = "[{\"type\": \"RATE\", \"text\": \"5%\", \"claim\": \"It is 5%.\"}]"
    
    llm_mock.chat.side_effect = [
        {"text": first_response},
        {"text": second_response}
    ]
    
    extractor = LLMAtomExtractor(llm=llm_mock, prompt_path="atoms/prompts/extract_v1.txt")
    
    # We must mock load_experiment since it's used inside _request_json
    import common.experiment
    orig_load = common.experiment.load_experiment
    common.experiment.load_experiment = lambda: {"llm_extract": {"json_mode": True}}
    
    try:
        atoms = extractor.extract("Some generated answer")
        
        # We should have exactly 1 atom after the repair
        assert len(atoms) == 1
        assert atoms[0].type == "RATE"
        assert atoms[0].text == "5%"
        assert extractor.parse_failures == 0
        
        # Check that the repair prompt included the error message from JSONDecodeError
        assert llm_mock.chat.call_count == 2
        args, kwargs = llm_mock.chat.call_args_list[1]
        user_prompt = kwargs['user']
        assert "failed to parse with the following error" in user_prompt
        assert "Expecting ',' delimiter" in user_prompt or "Extra data" in user_prompt or "Expecting property name" in user_prompt
    finally:
        common.experiment.load_experiment = orig_load

def test_json_parse_failure():
    llm_mock = MagicMock()
    # Both responses are invalid
    llm_mock.chat.return_value = {"text": "totally not json"}
    
    extractor = LLMAtomExtractor(llm=llm_mock, prompt_path="atoms/prompts/extract_v1.txt")
    
    import common.experiment
    orig_load = common.experiment.load_experiment
    common.experiment.load_experiment = lambda: {"llm_extract": {"json_mode": True}}
    
    try:
        atoms = extractor.extract("Another generated answer")
        
        assert len(atoms) == 0
        assert extractor.parse_failures == 1
    finally:
        common.experiment.load_experiment = orig_load
