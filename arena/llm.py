"""Minimal client for any OpenAI-compatible chat-completions API.

Built to run as a service:
- credentials come from the caller (env vars / UI), never from a plaintext file;
- the rate limiter only waits when calls are actually too close together;
- 429 / 5xx / network errors are retried with exponential backoff;
- optional JSON mode, with an automatic fallback for servers that reject it;
- thread-safe, because the web app can serve several users at once.
"""

import datetime as dt
import threading
import time
from typing import Optional

import requests

RETRY_STATUS = {429, 500, 502, 503, 504}


class LLMError(RuntimeError):
    """Raised when the LLM API cannot produce an answer."""


def chat_completions_url(base_url: str) -> str:
    """Accept either an API base (`.../v1`) or the full endpoint (`.../v1/chat/completions`)."""
    url = base_url.rstrip("/")
    return url if url.endswith("/chat/completions") else f"{url}/chat/completions"


class LLMClient:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model_name: str,
        temperature: float = 0.2,
        max_queries_per_minute: int = 30,
        max_tokens_per_day: int = 500_000,
        json_mode: bool = True,
        timeout: float = 120,
        max_retries: int = 4,
        extra_body: Optional[dict] = None,
    ):
        if not api_key:
            raise LLMError(
                "No LLM API key configured. Set LLM_API_KEY (e.g. a free key from https://console.groq.com)."
            )
        self.url = chat_completions_url(base_url)
        self.model_name = model_name
        self.temperature = temperature
        self.min_interval = 60.0 / max_queries_per_minute if max_queries_per_minute > 0 else 0.0
        self.max_tokens_per_day = max_tokens_per_day
        self.json_mode = json_mode
        self.timeout = timeout
        self.max_retries = max_retries
        self.extra_body = extra_body or {}  # provider-specific fields, e.g. {"reasoning_effort": "low"}
        self.headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

        self._lock = threading.Lock()
        self._last_call = 0.0
        self._day = dt.date.today()
        self.total_tokens_used = 0

    # ---------- budget & pacing ----------
    def _reserve(self, estimated_tokens: int) -> None:
        with self._lock:
            today = dt.date.today()
            if today != self._day:
                self._day, self.total_tokens_used = today, 0
            if self.max_tokens_per_day and self.total_tokens_used + estimated_tokens > self.max_tokens_per_day:
                raise LLMError("Daily token budget exceeded (LLM_MAX_TOKENS_PER_DAY). Try again tomorrow.")
            wait = self._last_call + self.min_interval - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()

    def _record_usage(self, result: dict, estimated_tokens: int) -> None:
        used = (result.get("usage") or {}).get("total_tokens") or estimated_tokens
        with self._lock:
            self.total_tokens_used += used

    # ---------- public API ----------
    def query(self, user_input: str, system_message: str) -> str:
        """Send one system+user exchange and return the assistant text ('' if none)."""
        estimated = (len(user_input) + len(system_message)) // 4  # ~4 chars per token
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_input},
            ],
            "temperature": self.temperature,
            **self.extra_body,
        }

        for attempt in range(self.max_retries + 1):
            if self.json_mode:
                payload["response_format"] = {"type": "json_object"}
            else:
                payload.pop("response_format", None)

            self._reserve(estimated)
            try:
                resp = requests.post(self.url, headers=self.headers, json=payload, timeout=self.timeout)
            except requests.RequestException as e:
                if attempt == self.max_retries:
                    raise LLMError(f"LLM request failed: {e}") from e
                time.sleep(2**attempt)
                continue

            if resp.status_code == 400 and self.json_mode:
                # Some servers/models do not support response_format: retry once without it.
                self.json_mode = False
                continue
            if resp.status_code in RETRY_STATUS and attempt < self.max_retries:
                try:
                    delay = float(resp.headers.get("retry-after"))
                except (TypeError, ValueError):
                    delay = 2**attempt
                time.sleep(delay)
                continue
            if resp.status_code in (401, 403):
                raise LLMError("The LLM API rejected the API key (HTTP %d)." % resp.status_code)
            if not resp.ok:
                raise LLMError(f"LLM API error HTTP {resp.status_code}: {resp.text[:300]}")

            result = resp.json()
            self._record_usage(result, estimated)
            choices = result.get("choices") or []
            return (choices[0].get("message", {}).get("content") or "") if choices else ""

        raise LLMError("LLM API kept failing after retries.")
