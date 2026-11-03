"""SFT format warm-up with LoRA, in plain PyTorch.

Input is the per-turn data from make_sft_data.py --format turns. Each row is a
context window (exactly what the agent will see at eval time) and the next
action. Loss is on the action tokens only.

Fits on an 8 GB laptop GPU for a 1.5B model: bf16 weights, LoRA on every
linear layer, gradient checkpointing, batch size 1 with accumulation, and
logits computed only for the completion. A full 152k-vocab logit tensor over
a 4k-token context would be several GB on its own.

    python examples/make_sft_data.py --format turns --difficulty all --episodes 150 \
        --max-turns-per-episode 12 --history-turns 40 --out data/sft_turns.jsonl
    python examples/sft_lora.py --data data/sft_turns.jsonl --out runs/sft-1.5b
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


def encode(tok, row):
    prompt = tok.apply_chat_template(row["prompt"], add_generation_prompt=True, tokenize=False)
    completion = row["completion"] + "<|im_end|>"  # Qwen chat format; teach the model to stop
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    c_ids = tok(completion, add_special_tokens=False)["input_ids"]
    return p_ids, c_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--data", type=Path, default=Path("data/sft_turns.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("runs/sft-1.5b"))
    ap.add_argument("--max-rows", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=6144)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    random.seed(a.seed)
    torch.manual_seed(a.seed)
    rows = [json.loads(l) for l in a.data.open()]
    random.shuffle(rows)
    rows = rows[: a.max_rows]

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).cuda()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    ))
    model.print_trainable_parameters()

    data = []
    for r in rows:
        p, c = encode(tok, r)
        if len(p) + len(c) <= a.max_len:
            data.append((p, c))
    print(f"{len(data)} examples (dropped {len(rows) - len(data)} over {a.max_len} tokens)")

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    total_steps = math.ceil(len(data) * a.epochs / a.grad_accum)
    warmup = max(1, total_steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warmup)
                                              * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total_steps))))
    log = (a.out / "train_log.jsonl")
    a.out.mkdir(parents=True, exist_ok=True)
    logf = log.open("w")

    model.train()
    step, running, t0 = 0, [], time.time()
    for epoch in range(a.epochs):
        random.shuffle(data)
        for i, (p, c) in enumerate(data):
            ids = torch.tensor([p + c], device="cuda")
            # logits for the last len(c)+1 positions: position t predicts token t+1
            logits = model(input_ids=ids, logits_to_keep=len(c) + 1).logits[0, :-1].float()
            loss = torch.nn.functional.cross_entropy(logits, ids[0, -len(c):])
            (loss / a.grad_accum).backward()
            running.append(loss.item())
            if (i + 1) % a.grad_accum == 0 or i == len(data) - 1:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % 10 == 0 or step == total_steps:
                    rec = {"step": step, "of": total_steps, "loss": round(sum(running) / len(running), 4),
                           "lr": sched.get_last_lr()[0], "elapsed_s": round(time.time() - t0),
                           "max_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
                    print(json.dumps(rec), flush=True)
                    logf.write(json.dumps(rec) + "\n")
                    logf.flush()
                    running = []
    model.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    print(f"saved adapter to {a.out}")


if __name__ == "__main__":
    main()
