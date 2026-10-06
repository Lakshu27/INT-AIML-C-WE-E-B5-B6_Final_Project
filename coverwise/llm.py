"""Thin LLM wrapper supporting Groq and Google Gemini.

`LLMClient.available` is False when no key/provider is configured; every
module that uses the LLM has a deterministic rule-based fallback, so the
app still works (with less fluent text) in offline mode.
"""
from __future__ import annotations

import json
import logging
import re
import time

from . import config

log = logging.getLogger(__name__)


class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(self, provider: str | None = None, api_key: str | None = None,
                 model: str | None = None, temperature: float | None = None):
        self.provider = (provider or config.LLM_PROVIDER or "none").lower()
        self.temperature = config.LLM_TEMPERATURE if temperature is None else temperature
        self.client = None
        self.model = model
        self.calls = 0
        if self.provider == "groq":
            key = api_key or config.GROQ_API_KEY
            self.model = model or config.GROQ_MODEL
            if key:
                try:
                    from groq import Groq
                    self.client = Groq(api_key=key)
                except ImportError:
                    log.warning("groq package not installed")
        elif self.provider == "gemini":
            key = api_key or config.GEMINI_API_KEY
            self.model = model or config.GEMINI_MODEL
            if key:
                try:
                    from google import genai
                    self.client = genai.Client(api_key=key)
                except ImportError:
                    log.warning("google-genai package not installed")

    @property
    def available(self) -> bool:
        return self.client is not None

    @property
    def label(self) -> str:
        return f"{self.provider}:{self.model}" if self.available else "offline (rule-based)"

    # ------------------------------------------------------------------ core
    def chat(self, system: str, user: str, json_mode: bool = False,
             max_tokens: int = 2048, retries: int = 3) -> str:
        if not self.available:
            raise LLMError("No LLM configured")
        last_exc = None
        attempt = 0
        while attempt < retries:
            attempt += 1
            try:
                self.calls += 1
                if self.provider == "groq":
                    kwargs = dict(model=self.model, temperature=self.temperature,
                                  max_tokens=max_tokens,
                                  messages=[{"role": "system", "content": system},
                                            {"role": "user", "content": user}])
                    if json_mode:
                        kwargs["response_format"] = {"type": "json_object"}
                    if "gpt-oss" in (self.model or ""):
                        # reasoning model: keep thinking short so the answer fits in max_tokens
                        kwargs["reasoning_effort"] = "low"
                        kwargs["max_tokens"] = max(max_tokens, 1024)
                    resp = self.client.chat.completions.create(**kwargs)
                    return resp.choices[0].message.content or ""
                # gemini
                from google.genai import types
                cfg = types.GenerateContentConfig(
                    system_instruction=system, temperature=self.temperature,
                    max_output_tokens=max_tokens,
                    response_mime_type="application/json" if json_mode else "text/plain",
                )
                resp = self.client.models.generate_content(model=self.model, contents=user, config=cfg)
                return resp.text or ""
            except Exception as exc:
                last_exc = exc
                msg = str(exc)
                low = msg.lower()
                # daily quota exhausted -> switch once to the fallback model (separate quota), no waiting
                if "429" in msg and ("per day" in low or "tpd" in low or "rpd" in low):
                    fb = config.GROQ_FALLBACK_MODEL if self.provider == "groq" else ""
                    if fb and fb != self.model:
                        log.warning("Daily limit reached for %s - switching to %s", self.model, fb)
                        self.model = fb
                        attempt -= 1  # the switch itself is not a failed attempt
                        continue
                    break
                # permanent errors (bad key / unknown model) -> stop immediately
                if any(code in msg for code in ("401", "403", "404", "invalid_api_key", "model_not_found")):
                    break
                # per-minute rate limit: wait as long as the API asks (max 20 s), else exponential backoff
                m = re.search(r"try again in ([\d.]+)s", low)
                wait = min(float(m.group(1)) + 0.5, 20) if m else 2 ** attempt
                log.warning("LLM call failed (%s). Retrying in %.1fs", msg[:160], wait)
                time.sleep(wait)
        raise LLMError(f"LLM call failed after {attempt} attempt(s) on {self.model}: {last_exc}")

    def chat_json(self, system: str, user: str, max_tokens: int = 2048) -> dict:
        raw = self.chat(system, user + "\n\nReturn ONLY valid JSON.", json_mode=True,
                        max_tokens=max_tokens)
        return parse_json(raw)


def parse_json(raw: str) -> dict:
    """Robust JSON parsing: strips ```json fences and trailing prose."""
    text = re.sub(r"```(?:json)?", "", raw).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    raise LLMError(f"Could not parse JSON from LLM output: {raw[:300]}")