"""
llm_client.py — Unified LLM client with per-stage provider selection.

Three stages use LLM calls:
  - data_prep:  augmentation + system prompt synthesis  (high volume, cost-sensitive)
  - agent:      hyperparameter search decisions          (needs strong reasoning)
  - evaluator:  Claude-as-judge for semantic matching    (moderate volume)

Each stage can independently use Claude or Ollama, giving the user
full flexibility: completely local, completely online, or hybrid.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Optional

import requests

# ── Stage names ──────────────────────────────────────────────────────────────

STAGE_DATA_PREP = "data_prep"
STAGE_AGENT = "agent"
STAGE_EVALUATOR = "evaluator"
ALL_STAGES = [STAGE_DATA_PREP, STAGE_AGENT, STAGE_EVALUATOR]


# ── Per-stage config ─────────────────────────────────────────────────────────


@dataclass
class StageConfig:
    """Provider config for a single stage."""

    provider: str = "claude"  # "claude" | "ollama"
    ollama_url: str = "http://127.0.0.1:11434"
    ollama_model: str = ""  # e.g. "llama3.1", "mistral"

    def is_ollama(self) -> bool:
        return self.provider == "ollama"

    def is_claude(self) -> bool:
        return self.provider == "claude"

    @property
    def label(self) -> str:
        if self.is_ollama():
            return f"Ollama ({self.ollama_model})"
        return "Claude API"


@dataclass
class LLMConfig:
    """Full configuration — one StageConfig per stage."""

    data_prep: StageConfig = field(default_factory=StageConfig)
    agent: StageConfig = field(default_factory=StageConfig)
    evaluator: StageConfig = field(default_factory=StageConfig)

    def get_stage(self, stage: str) -> StageConfig:
        return getattr(self, stage)

    def set_stage(self, stage: str, cfg: StageConfig) -> None:
        setattr(self, stage, cfg)


# Module-level config — set once at app startup
_active_config = LLMConfig()


def configure(config: LLMConfig) -> None:
    """Set the active LLM config."""
    global _active_config
    _active_config = config


def get_config() -> LLMConfig:
    """Return the active LLM config."""
    return _active_config


def get_stage_config(stage: str) -> StageConfig:
    """Return the config for a specific stage."""
    return _active_config.get_stage(stage)


# ── Generation ───────────────────────────────────────────────────────────────


def generate(
    prompt: str,
    stage: str = STAGE_DATA_PREP,
    model_hint: str = "fast",
    max_tokens: int = 4096,
    temperature: float = 0.9,
) -> str:
    """Generate text using the provider configured for the given stage.

    Args:
        prompt: The user message to send.
        stage: Which pipeline stage is calling — determines provider.
        model_hint: "fast" (Haiku-class) or "smart" (Sonnet-class).
                    Only used when provider is Claude.
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.

    Returns:
        The generated text string.
    """
    cfg = _active_config.get_stage(stage)

    if cfg.is_ollama():
        return _generate_ollama(prompt, cfg, max_tokens, temperature)
    else:
        return _generate_claude(prompt, model_hint, max_tokens, temperature)


def _generate_claude(
    prompt: str,
    model_hint: str,
    max_tokens: int,
    temperature: float,
) -> str:
    """Call Anthropic API."""
    import anthropic

    model_map = {
        "fast": "claude-haiku-4-5-20251001",
        "smart": "claude-sonnet-4-20250514",
    }
    model = model_map.get(model_hint, model_map["fast"])

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()


def _validate_ollama_url(url: str) -> str:
    """Reject non-HTTP schemes to prevent SSRF via user-supplied Ollama URL."""
    from urllib.parse import urlparse

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Ollama URL must use http or https scheme, got: {parsed.scheme!r}")
    return url


def _generate_ollama(
    prompt: str,
    cfg: StageConfig,
    max_tokens: int,
    temperature: float,
) -> str:
    """Call Ollama API (/api/generate endpoint)."""
    _validate_ollama_url(cfg.ollama_url)
    url = cfg.ollama_url.rstrip("/") + "/api/generate"

    payload = {
        "model": cfg.ollama_model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }

    resp = requests.post(url, json=payload, timeout=300)
    resp.raise_for_status()
    data = resp.json()
    return data.get("response", "").strip()


# ── Ollama discovery helpers ─────────────────────────────────────────────────


def list_ollama_models(ollama_url: str = "http://127.0.0.1:11434") -> list[str]:
    """Fetch available model names from an Ollama server."""
    try:
        _validate_ollama_url(ollama_url)
        url = ollama_url.rstrip("/") + "/api/tags"
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        models = [m["name"] for m in data.get("models", [])]
        return sorted(models)
    except Exception:
        return []


def check_ollama_server(ollama_url: str = "http://127.0.0.1:11434") -> bool:
    """Return True if the Ollama server is reachable."""
    try:
        _validate_ollama_url(ollama_url)
        resp = requests.get(ollama_url.rstrip("/") + "/api/tags", timeout=3)
        return resp.status_code == 200
    except Exception:
        return False
