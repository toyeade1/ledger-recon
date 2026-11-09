"""Merge a LoRA adapter into its base model and save a plain HF checkpoint.

Serving engines like llama.cpp and vLLM are much faster than transformers'
generate() for multi-turn rollouts. Merge, convert, then eval the base and
the fine-tuned model through the same server so the comparison is fair.

    python examples/merge_lora.py --adapter runs/sft-1.5b --out runs/sft-1.5b-merged
    python llama.cpp/convert_hf_to_gguf.py runs/sft-1.5b-merged --outtype f16 --outfile sft-1.5b-f16.gguf
"""

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter", help="omit to just save the base model locally")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16)
    if a.adapter:
        model = PeftModel.from_pretrained(model, a.adapter).merge_and_unload()
    model.save_pretrained(a.out)
    AutoTokenizer.from_pretrained(a.adapter or a.model).save_pretrained(a.out)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
