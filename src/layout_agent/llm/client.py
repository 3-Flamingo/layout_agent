"""OpenAI-compatible LLM client, mirroring businessplatform_indi LlmClient."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import httpx


class LlmClient:
    """百炼 / 私有网关 Chat Completions（与交付平台 LlmClient 对齐）。"""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str = "deepseek-v4-pro",
        timeout_s: float = 60.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_s = timeout_s

    def is_configured(self) -> bool:
        return bool(self.api_key and self.base_url)

    def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0.2,
        enable_thinking: bool = False,
    ) -> Dict[str, Any]:
        if not self.is_configured():
            raise RuntimeError("LLM 未配置：请设置 DASHSCOPE_API_KEY 与 OPENAI_BASE_URL")

        url = f"{self.base_url}/chat/completions"
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": False,
        }
        # 布局决策需要稳定 JSON，默认关闭深度思考以降低延迟
        payload["enable_thinking"] = bool(enable_thinking)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        with httpx.Client(timeout=self.timeout_s) as client:
            resp = client.post(url, headers=headers, content=json.dumps(payload))
        if resp.status_code >= 300:
            raise RuntimeError(f"LLM HTTP {resp.status_code}: {resp.text[:500]}")
        data = resp.json()
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("LLM 返回为空")
        message = choices[0].get("message") or {}
        return message

    def chat_text(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0.2,
        enable_thinking: bool = False,
    ) -> str:
        message = self.chat_completion(messages, temperature, enable_thinking)
        content = message.get("content")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(str(block.get("text") or ""))
            return "\n".join(parts).strip()
        return ""
