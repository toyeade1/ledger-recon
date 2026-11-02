"""Run any OpenAI-compatible chat model as the agent.

Works with OpenAI, OpenRouter, vLLM, SGLang, Ollama, LM Studio: anything that
serves /v1/chat/completions. Point it with --base-url and --model.

    # local, e.g. Ollama
    python -m ledger_recon.eval --agent llm --base-url http://localhost:11434/v1 --model qwen3:4b
    # hosted
    OPENAI_API_KEY=... python -m ledger_recon.eval --agent llm --model gpt-5-mini

Context management: Wikipedia-style envs blow up context because each page is
long. Here each observation is short, but a 90-turn episode still adds up, so
we keep the system prompt + first observation + the last `history_turns`
exchanges. That truncation is a design choice worth experimenting with.
"""

from __future__ import annotations

from typing import Any

from .env import SYSTEM_PROMPT


class LLMAgent:
    def __init__(self, model: str, client: Any = None, base_url: str | None = None, api_key: str | None = None,
                 temperature: float = 0.7, max_tokens: int = 1024, history_turns: int = 12):
        if client is None:
            from openai import OpenAI  # optional dependency
            client = OpenAI(base_url=base_url, api_key=api_key or "not-needed-for-local")
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.history_turns = history_turns
        self.messages: list[dict[str, str]] = []

    def act(self, obs: str, info: dict) -> str:
        if not self.messages:
            self.messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": obs}]
        else:
            self.messages.append({"role": "user", "content": obs})
        head, tail = self.messages[:2], self.messages[2:]
        window = head + tail[-2 * self.history_turns:]
        resp = self.client.chat.completions.create(
            model=self.model, messages=window, temperature=self.temperature, max_tokens=self.max_tokens,
        )
        text = resp.choices[0].message.content or ""
        self.messages.append({"role": "assistant", "content": text})
        return text
