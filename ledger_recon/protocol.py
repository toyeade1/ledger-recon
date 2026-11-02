"""Text action protocol: how a language model's output becomes a tool call.

Tokens in, tokens out. The model writes optional reasoning in <think> tags and
exactly one tool call as JSON inside <tool> tags:

    <think>TXN-41822 says REF INV-2026-0412 and the amount matches.</think>
    <tool>{"name": "match", "args": {"transaction_id": "TXN-41822", "invoice_ids": ["INV-2026-0412"]}}</tool>

Anything that doesn't parse is an invalid turn. The env tells the model what
went wrong and lets it keep going (with a small penalty) instead of ending the
episode. Small models break format a lot early on; killing the episode on the
first mistake starves them of learning signal.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

TOOL_RE = re.compile(r"<tool>\s*(\{.*?\})\s*</tool>", re.DOTALL)


@dataclass
class ParsedAction:
    name: str | None
    args: dict[str, Any]
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def parse(text: str) -> ParsedAction:
    found = TOOL_RE.findall(text)
    if not found:
        return ParsedAction(None, {}, 'No tool call found. End your reply with <tool>{"name": ..., "args": {...}}</tool>.')
    if len(found) > 1:
        return ParsedAction(None, {}, "Exactly one <tool> call per turn. You sent more than one.")
    try:
        obj = json.loads(found[0])
    except json.JSONDecodeError as e:
        return ParsedAction(None, {}, f"Tool call is not valid JSON: {e.msg}.")
    if not isinstance(obj, dict) or not isinstance(obj.get("name"), str):
        return ParsedAction(None, {}, 'Tool call must be an object with a string "name".')
    args = obj.get("args", {})
    if not isinstance(args, dict):
        return ParsedAction(None, {}, '"args" must be an object.')
    return ParsedAction(obj["name"], args)


def render(name: str, args: dict[str, Any] | None = None, think: str | None = None) -> str:
    """Inverse of parse(). Handy for scripted agents and SFT data generation."""
    body = json.dumps({"name": name, "args": args or {}})
    prefix = f"<think>{think}</think>\n" if think else ""
    return f"{prefix}<tool>{body}</tool>"
