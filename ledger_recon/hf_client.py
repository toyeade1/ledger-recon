"""Run a local Hugging Face model (optionally with a LoRA adapter) as the agent.

It mimics the small slice of the OpenAI client that LLMAgent calls,
`client.chat.completions.create(...)`, so the eval loop doesn't change. Use it
to compare a base model and its fine-tuned adapter on identical seeds.

    python -m ledger_recon.eval --agent llm --backend hf --model Qwen/Qwen2.5-1.5B-Instruct \
        --adapter runs/sft-1.5b --history-turns 40 --difficulty easy --episodes 20

Needs torch, transformers and (for --adapter) peft. Not a core dependency.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any


class HFChatClient:
    def __init__(self, model: str, adapter: str | None = None, device: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.tok = AutoTokenizer.from_pretrained(adapter or model)
        self.model = AutoModelForCausalLM.from_pretrained(model, dtype=dtype).to(self.device)
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter).merge_and_unload()
        self.model.eval()
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, model: str, messages: list[dict[str, str]], temperature: float = 0.7,
                max_tokens: int = 512, **_: Any):
        torch = self.torch
        ids = self.tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        if not isinstance(ids, torch.Tensor):  # newer transformers return a BatchEncoding
            ids = ids["input_ids"]
        ids = ids.to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                ids, max_new_tokens=max_tokens, do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None, top_p=None, top_k=None,
                pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id,
            )
        text = self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])
