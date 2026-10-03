import os
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional
import openai
from tenacity import retry, wait_exponential_jitter, stop_after_attempt, retry_if_exception, before_sleep_log
import logging

from common.rate_limit import get_bucket

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)
except ImportError:
    pass

logger = logging.getLogger("llm_client")

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", ".llm_cache")

def get_prompt_hash(model: str, system: str, user: str, kwargs: dict) -> str:
    """Generates a stable hash for a prompt and its parameters."""
    payload = {
        "model": model,
        "system": system,
        "user": user,
        "kwargs": {k: v for k, v in kwargs.items() if k != "api_key"} 
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

def is_retryable_error(exception):
    if isinstance(exception, openai.RateLimitError):
        return True
    if isinstance(exception, openai.APIStatusError):
        return exception.status_code >= 500 or exception.status_code == 429
    if isinstance(exception, openai.APIConnectionError):
        return True
    return False

class LLMClient:
    """
    OpenAI-compatible client wrapper for vLLM and API models, 
    with retries and a disk cache.
    """
    def __init__(
        self,
        base_url: str = None,
        api_key: str = None,
        default_model: str = None,
        config: dict = None,
    ):
        if config:
            base_url = config.get("base_url", base_url)
            default_model = config.get("model", default_model)
            api_key_env = config.get("api_key_env")
            if api_key_env:
                api_key = os.getenv(api_key_env, api_key)

        base_url = base_url or os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
        api_key  = api_key  or os.getenv("LLM_API_KEY",  "ollama")
        default_model = default_model or os.getenv("LLM_MODEL", "qwen2.5:7b-instruct-q4_K_M")
        
        provider = "ollama"
        if "api.openai.com" in base_url: provider = "openai"
        elif "api.groq.com" in base_url: provider = "groq"
        elif "api.anthropic.com" in base_url: provider = "anthropic"
        elif base_url and "localhost" not in base_url: provider = "custom"
        
        print(f"LLMClient initialized - Provider: {provider}, Model: {default_model}")
        
        self.client = openai.OpenAI(base_url=base_url, api_key=api_key)
        self.default_model = default_model
        os.makedirs(CACHE_DIR, exist_ok=True)
        
    def _read_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        path = os.path.join(CACHE_DIR, f"{cache_key}.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None
        
    def _write_cache(self, cache_key: str, data: Dict[str, Any]):
        path = os.path.join(CACHE_DIR, f"{cache_key}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    @retry(
        wait=wait_exponential_jitter(initial=1, max=60, jitter=1),
        stop=stop_after_attempt(int(os.getenv("LLM_MAX_RETRIES", "6"))),
        retry=retry_if_exception(is_retryable_error),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def chat(self, user: str, system: Optional[str] = None, model: Optional[str] = None, logprobs: bool = False, temperature: float = 0.0, **kwargs) -> Dict[str, Any]:
        """
        Sends a chat request with caching and retries.
        """
        model = model or self.default_model
        system_msg = system or "You are a helpful assistant."
        
        cache_key = get_prompt_hash(model, system_msg, user, {"temperature": temperature, "logprobs": logprobs, **kwargs})
        
        cached = self._read_cache(cache_key)
        if cached:
            cached = dict(cached)
            cached["cache_hit"] = True
            return cached

        get_bucket().acquire(1.0)

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user}
        ]
        
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            logprobs=logprobs,
            **kwargs
        )
        
        result = {
            "text": response.choices[0].message.content,
            "model": response.model,
            "cache_hit": False,
        }
        
        logger.info(f"LLM Generation completed - Model: {model}, Prompt Hash: {cache_key}")
        
        if logprobs and response.choices[0].logprobs:
            result["token_logprobs"] = [
                token.logprob for token in response.choices[0].logprobs.content
            ]
            
        self._write_cache(cache_key, result)
        return result
