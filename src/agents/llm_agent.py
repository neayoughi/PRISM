from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict

try:
    from langchain_core.messages import SystemMessage, HumanMessage
except Exception:
    @dataclass
    class SystemMessage:  # type: ignore[override]
        content: str

    @dataclass
    class HumanMessage:  # type: ignore[override]
        content: str

from .llm_utils import get_llm, llm_call


@dataclass
class AgentConfig:
    model: str
    max_retries: int = 2


def call_classifier(
    cfg: AgentConfig,
    system_prompt: str,
    user_prompt: str,
) -> str:
    llm = get_llm(cfg.model)
    msgs = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    return llm_call(llm, msgs, use_logprobs=False)


def parse_json_strict(text: str) -> Dict[str, Any]:
    t = text.strip()
    if t.startswith("```"):
        lines = t.splitlines()
        if lines:
            lines = lines[1:]
        while lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        t = "\n".join(lines).strip()
    return json.loads(t)


def repair_json(cfg: AgentConfig, system_prompt: str, bad_text: str, schema_text: str) -> str:
    repair_prompt = (
        "Return valid JSON only. "
        "Fix the JSON so it matches this schema. "
        "Do not add extra keys.\n\n"
        "SCHEMA\n" + schema_text + "\n\n"
        "BAD_OUTPUT\n" + bad_text
    )
    llm = get_llm(cfg.model)
    msgs = [SystemMessage(content=system_prompt), HumanMessage(content=repair_prompt)]
    return llm_call(llm, msgs, use_logprobs=False)
