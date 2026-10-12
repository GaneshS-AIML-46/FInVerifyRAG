"""Multi-provider LLM client for RegGuard pipeline.

Supported providers: openai, anthropic, groq (openai-compatible), ollama (openai-compatible).
No silent fallbacks. Missing config raises clear errors.
"""
from __future__ import annotations

import json
import logging
import os
import time
import hashlib
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from common.errors import BudgetExceededError, LLMError, PriceMissingError
from common.rate_limit import get_bucket

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)
except ImportError:
    pass

logger = logging.getLogger("llm_client")

CACHE_DIR = Path(__file__).resolve().parent.parent / ".llm_cache"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# Per-role accumulated cost in memory
_role_costs: Dict[str, float] = {}


def _cache_key(model: str, system: str, user: str, kwargs: dict) -> str:
    payload = {"model": model, "system": system, "user": user,
                "kwargs": {k: v for k, v in kwargs.items() if k != "api_key"}}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _is_retryable(exc: Exception) -> bool:
    try:
        import openai
        if isinstance(exc, openai.RateLimitError):
            return True
        if isinstance(exc, openai.APIStatusError):
            return exc.status_code in (429, 500, 502, 503, 529)
        if isinstance(exc, openai.APIConnectionError):
            return True
    except ImportError:
        pass
    try:
        import anthropic
        if isinstance(exc, anthropic.RateLimitError):
            return True
        if isinstance(exc, anthropic.APIStatusError):
            return exc.status_code in (429, 500, 502, 503, 529)
        if isinstance(exc, anthropic.APIConnectionError):
            return True
        if hasattr(anthropic, 'OverloadedError') and isinstance(exc, anthropic.OverloadedError):
            return True
    except ImportError:
        pass
    return False


class LLMClient:
    """Role-aware multi-provider LLM client.

    Args:
        role: one of question_drafting, generator, extractor, judge
        cfg: full experiment.yaml dict
        check_prices: if True (default), raises PriceMissingError when price is null
    """

    def __init__(self, role: str, cfg: dict, check_prices: bool = True):
        roles_cfg = cfg.get("roles")
        if not roles_cfg:
            raise RuntimeError("experiment.yaml is missing 'roles' section")
        if role not in roles_cfg:
            raise RuntimeError(f"Role '{role}' not found in cfg['roles']. Available: {list(roles_cfg)}")

        role_cfg = roles_cfg[role]
        self.role = role
        self.provider: str = role_cfg["provider"]
        self.model: str = role_cfg["model"]
        self.max_tokens: int = int(role_cfg.get("max_tokens", 1024))
        self.temperature: float = float(role_cfg.get("temperature", 0.0))
        api_key_env: str = role_cfg["api_key_env"]

        # Resolve API key
        api_key = os.getenv(api_key_env)
        if not api_key and self.provider != "ollama":
            raise RuntimeError(
                f"[{role}] API key env var '{api_key_env}' is not set. "
                f"Export it before running."
            )
        self._api_key = api_key  # stored but never logged

        # Price check
        if check_prices:
            prices = cfg.get("llm_prices", {})
            model_prices = prices.get(self.model, {})
            if model_prices.get("input") is None or model_prices.get("output") is None:
                raise PriceMissingError(
                    f"Price missing for model '{self.model}'. "
                    f"Set it in configs/experiment.yaml before running."
                )
        self._prices = cfg.get("llm_prices", {}).get(self.model, {})

        # Budget caps
        self._budget_caps = cfg.get("budget_caps", {})

        print(f"[{role}] provider={self.provider} model={self.model}")

        # Build underlying SDK client
        self._anthropic_client = None
        self._openai_client = None

        if self.provider == "anthropic":
            import anthropic as _anthropic
            self._anthropic_client = _anthropic.Anthropic(api_key=self._api_key)
        else:
            import openai as _openai
            base_url_map = {
                "openai": "https://api.openai.com/v1",
                "groq": "https://api.groq.com/openai/v1",
                "ollama": "http://localhost:11434/v1",
            }
            base_url = role_cfg.get("base_url") or base_url_map.get(self.provider, "http://localhost:11434/v1")
            self._openai_client = _openai.OpenAI(
                api_key=self._api_key or "ollama",
                base_url=base_url,
            )

        CACHE_DIR.mkdir(exist_ok=True)

    def _read_cache(self, key: str) -> Optional[Dict[str, Any]]:
        p = CACHE_DIR / f"{key}.json"
        if p.exists():
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        return None

    def _write_cache(self, key: str, data: Dict[str, Any]) -> None:
        p = CACHE_DIR / f"{key}.json"
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def _compute_cost(self, input_tokens: int, output_tokens: int) -> float:
        if not self._prices:
            return 0.0
        inp_price = self._prices.get("input")
        out_price = self._prices.get("output")
        if inp_price is None or out_price is None:
            return 0.0
        return (input_tokens * inp_price + output_tokens * out_price) / 1_000_000

    def _check_budget(self, role: str, additional_cost: float) -> None:
        cap = self._budget_caps.get(role)
        if cap is None:
            return
        current = _role_costs.get(role, 0.0)
        if current + additional_cost > cap:
            raise BudgetExceededError(
                f"Budget cap for role '{role}' exceeded: "
                f"spent ${current:.4f}, cap=${cap:.2f}, "
                f"new cost=${additional_cost:.4f}"
            )

    def chat(self, user: str, system: str = "", **kwargs) -> Dict[str, Any]:
        """Send a chat message, with caching, retries, and budget check."""
        key = _cache_key(self.model, system, user, {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            **kwargs,
        })

        cached = self._read_cache(key)
        if cached:
            result = dict(cached)
            result["cache_hit"] = True
            result["cost_usd"] = 0.0
            return result

        get_bucket().acquire(1.0)

        max_attempts = 6
        last_exc: Exception | None = None
        for attempt in range(max_attempts):
            try:
                if self.provider == "anthropic":
                    result = self._call_anthropic(user, system)
                else:
                    result = self._call_openai(user, system)

                cost = self._compute_cost(result["input_tokens"], result["output_tokens"])
                self._check_budget(self.role, cost)
                _role_costs[self.role] = _role_costs.get(self.role, 0.0) + cost
                result["cost_usd"] = cost
                result["cache_hit"] = False

                self._write_cache(key, result)
                return result

            except (BudgetExceededError,) as e:
                raise
            except Exception as e:
                if _is_retryable(e):
                    last_exc = e
                    wait = min(2 ** attempt + random.uniform(0, 1), 60)
                    logger.warning(f"[{self.role}] Retryable error (attempt {attempt+1}/{max_attempts}): {e}. Waiting {wait:.1f}s")
                    time.sleep(wait)
                    continue
                # Non-retryable: raise as LLMError with no fallback
                raise LLMError(f"[{self.role}] LLM call failed: {e}") from e

        raise LLMError(f"[{self.role}] Max retries ({max_attempts}) exceeded. Last error: {last_exc}")

    def _call_anthropic(self, user: str, system: str) -> Dict[str, Any]:
        # temperature is not accepted by extended-thinking models (claude-sonnet-4-6, claude-3-7, etc.)
        # For those models temperature is internally fixed at 1.0; omit it to stay compatible.
        create_kwargs: Dict[str, Any] = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system or "You are a helpful assistant.",
            messages=[{"role": "user", "content": user}],
        )
        # Only pass temperature for models that support it (non-extended-thinking)
        if self.temperature is not None and not any(
            tok in self.model for tok in ("sonnet-4-6", "sonnet-4-7", "opus-4", "3-7-sonnet")
        ):
            create_kwargs["temperature"] = self.temperature
        response = self._anthropic_client.messages.create(**create_kwargs)
        return {
            "text": response.content[0].text,
            "model": response.model,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }

    def _call_openai(self, user: str, system: str) -> Dict[str, Any]:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        response = self._openai_client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        usage = response.usage or type("U", (), {"prompt_tokens": 0, "completion_tokens": 0})()
        return {
            "text": response.choices[0].message.content,
            "model": response.model,
            "input_tokens": getattr(usage, "prompt_tokens", 0),
            "output_tokens": getattr(usage, "completion_tokens", 0),
        }

    def record_cost(self, role: str, result: Dict[str, Any]) -> None:
        """Append cost record to results/cost.jsonl."""
        RESULTS_DIR.mkdir(exist_ok=True)
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "role": role,
            "model": self.model,
            "input_tokens": result.get("input_tokens", 0),
            "output_tokens": result.get("output_tokens", 0),
            "cost_usd": result.get("cost_usd", 0.0),
            "cache_hit": result.get("cache_hit", False),
        }
        with open(RESULTS_DIR / "cost.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
