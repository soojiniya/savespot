"""Small replaceable LLM provider wrapper.

Uses OpenAI Responses API via HTTPS when configured. The rest of the
application only depends on this file's tiny interface, so another provider can
replace it later without changing API handlers.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
import requests


OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
logger = logging.getLogger(__name__)


def safe_openai_error_payload(response: requests.Response | None) -> dict[str, Any]:
    if response is None:
        return {
            "status": None,
            "code": None,
            "type": None,
            "message": None,
        }

    status = response.status_code
    code = None
    error_type = None
    message = None
    try:
        payload = response.json()
    except ValueError:
        payload = {}

    error = payload.get("error") if isinstance(payload, dict) else None
    if isinstance(error, dict):
        code = error.get("code")
        error_type = error.get("type")
        message = error.get("message")

    return {
        "status": status,
        "code": code,
        "type": error_type,
        "message": message,
    }


def log_openai_error(exc: Exception, model: str, response: requests.Response | None = None) -> None:
    details = safe_openai_error_payload(response)
    message = details["message"] or str(exc)
    logger.warning(
        "OpenAIError:\n"
        "exception_class=%s\n"
        "status=%s\n"
        "code=%s\n"
        "type=%s\n"
        "message=%s\n"
        "model=%s",
        exc.__class__.__name__,
        details["status"],
        details["code"],
        details["type"],
        message,
        model,
    )


class LlmClient:
    def __init__(self) -> None:
        load_dotenv(PROJECT_ROOT / ".env")
        self.api_key = os.getenv("OPENAI_API_KEY", "").strip()
        self.model = os.getenv("AI_ASSISTANT_MODEL", "gpt-4o-mini").strip()
        self.enabled = os.getenv("AI_ASSISTANT_ENABLED", "true").lower() == "true"

    @property
    def available(self) -> bool:
        return self.enabled and bool(self.api_key)

    def create_response(
        self,
        input_data: str | list[dict[str, Any]],
        instructions: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
        previous_response_id: str | None = None,
    ) -> dict[str, Any]:
        if not self.available:
            raise RuntimeError("LLM provider is disabled or not configured")
        payload: dict[str, Any] = {
            "model": self.model,
            "input": input_data,
        }
        if instructions:
            payload["instructions"] = instructions
        if tools:
            payload["tools"] = tools
        if tool_choice:
            payload["tool_choice"] = tool_choice
        if previous_response_id:
            payload["previous_response_id"] = previous_response_id
        try:
            response = requests.post(
                OPENAI_RESPONSES_URL,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                timeout=30,
            )
            response.raise_for_status()
        except requests.HTTPError as exc:
            log_openai_error(exc, self.model, exc.response)
            raise
        except requests.RequestException as exc:
            log_openai_error(exc, self.model)
            raise
        return response.json()

