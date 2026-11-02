"""Evaluate agents across seeds and difficulty levels.

    python -m ledger_recon.eval                          # all scripted baselines
    python -m ledger_recon.eval --agent heuristic --difficulty hard --episodes 50
    python -m ledger_recon.eval --agent llm --model qwen3:4b --base-url http://localhost:11434/v1 \
        --episodes 10 --rollouts-out runs/qwen3-4b.jsonl

Seeds 0-9999 are the train split by convention; eval uses seeds from 10000 so
you never report numbers on scenarios you trained on.
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

from .agents import AGENTS
from .env import LedgerReconEnv
from .rollout import run_episode

EVAL_SEED_OFFSET = 10_000


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--agent", default="all", help=f"one of {sorted(AGENTS)} or 'llm' or 'all'")
    p.add_argument("--difficulty", default="all", choices=["easy", "medium", "hard", "all"])
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--model", help="model name for --agent llm")
    p.add_argument("--base-url", help="OpenAI-compatible endpoint for --agent llm")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--rollouts-out", type=Path, help="write every trajectory as JSONL")
    a = p.parse_args(argv)

    names = list(AGENTS) if a.agent == "all" else [a.agent]
    levels = ["easy", "medium", "hard"] if a.difficulty == "all" else [a.difficulty]
    out = a.rollouts_out.open("w") if a.rollouts_out else None
    if out:
        a.rollouts_out.parent.mkdir(parents=True, exist_ok=True)

    print(f"{'agent':<12} {'level':<7} {'reward':>7} {'recon':>7} {'solved':>7} {'invalid/ep':>10} {'turns':>6}")
    for name in names:
        for level in levels:
            rows = []
            for i in range(a.episodes):
                seed = EVAL_SEED_OFFSET + i
                if name == "llm":
                    from .llm import LLMAgent
                    agent = LLMAgent(a.model, base_url=a.base_url, temperature=a.temperature)
                else:
                    agent = AGENTS[name](seed)
                env = LedgerReconEnv(level)
                ro = run_episode(env, agent, seed, name)
                rows.append((ro, env.state.invalid_actions, env.state.turns))
                if out:
                    out.write(ro.to_json() + "\n")
            print(f"{name:<12} {level:<7} "
                  f"{statistics.mean(r.reward for r, _, _ in rows):>7.3f} "
                  f"{statistics.mean(r.breakdown['reconciliation'] for r, _, _ in rows):>7.3f} "
                  f"{sum(r.breakdown['solved'] for r, _, _ in rows) / len(rows):>7.0%} "
                  f"{statistics.mean(n for _, n, _ in rows):>10.2f} "
                  f"{statistics.mean(t for _, _, t in rows):>6.1f}")
    if out:
        out.close()


if __name__ == "__main__":
    main()
