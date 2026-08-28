from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> Dict[str, Any]:
    load_dotenv(project_root() / ".env")
    cfg_path = Path(path) if path else project_root() / "config" / "default.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    # env overrides (align with businessplatform_indi)
    decide = cfg.setdefault("decide", {})
    if os.getenv("OPENAI_BASE_URL"):
        decide["base_url"] = os.getenv("OPENAI_BASE_URL")
    if os.getenv("OPENAI_MODEL"):
        decide["model"] = os.getenv("OPENAI_MODEL")
    if os.getenv("LAYOUT_AGENT_USE_LLM") in ("1", "true", "True"):
        decide["use_llm"] = True
    if os.getenv("LAYOUT_AGENT_USE_LLM") in ("0", "false", "False"):
        decide["use_llm"] = False

    api_key = (
        os.getenv("DASHSCOPE_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or ""
    ).strip()
    decide["api_key"] = api_key
    return cfg
