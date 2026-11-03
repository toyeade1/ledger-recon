"""Generate SFT warm-up data from the scripted solver.

The tic-tac-toe recipe in the talk: small models break the action format, so
before RL you do a short SFT pass to teach format and valid moves, then let RL
build the actual skill. Same idea here. Keep only fully solved episodes so you
don't bake in mistakes.

Two output formats:

  episodes  one chat transcript per episode (good for reading)
  turns     one example per agent turn, cut with the same context window the
            LLM agent uses at eval time (system + first obs + last N
            exchanges). Train on this one, otherwise the model learns from
            contexts it will never see.

    python examples/make_sft_data.py --episodes 200 --out data/sft.jsonl
    python examples/make_sft_data.py --format turns --difficulty all --episodes 300 --out data/sft_turns.jsonl
"""

import argparse
import json
import random
from pathlib import Path

from ledger_recon import SYSTEM_PROMPT, LedgerReconEnv
from ledger_recon.agents import HeuristicAgent
from ledger_recon.llm import context_window
from ledger_recon.protocol import render


def solved_transcript(difficulty: str, seed: int) -> list[dict] | None:
    env, agent = LedgerReconEnv(difficulty), HeuristicAgent(seed)
    obs, info = env.reset(seed=seed)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": obs}]
    while True:
        act = agent.act(obs, info)
        messages.append({"role": "assistant", "content": render(act["name"], act.get("args"))})
        obs, reward, term, trunc, info = env.step(act)
        if term or trunc:
            break
        messages.append({"role": "user", "content": obs})
    return messages if info["reward_breakdown"]["solved"] else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=200, help="episodes per difficulty level")
    p.add_argument("--difficulty", default="medium", choices=["easy", "medium", "hard", "all"])
    p.add_argument("--format", default="episodes", choices=["episodes", "turns"])
    p.add_argument("--history-turns", type=int, default=12, help="must match LLMAgent at eval time")
    p.add_argument("--max-turns-per-episode", type=int, default=0,
                   help="turns format: sample at most this many turns per episode (0 = all)")
    p.add_argument("--out", type=Path, default=Path("data/sft.jsonl"))
    a = p.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    levels = ["easy", "medium", "hard"] if a.difficulty == "all" else [a.difficulty]
    rng = random.Random(0)
    kept = rows = 0
    with a.out.open("w") as f:
        for level in levels:
            for seed in range(a.episodes):  # train seeds only (< 10000)
                messages = solved_transcript(level, seed)
                if messages is None:
                    continue
                kept += 1
                if a.format == "episodes":
                    f.write(json.dumps({"messages": messages, "seed": seed, "difficulty": level}) + "\n")
                    rows += 1
                    continue
                idx = [i for i, m in enumerate(messages) if m["role"] == "assistant"]
                if a.max_turns_per_episode and len(idx) > a.max_turns_per_episode:
                    idx = sorted(rng.sample(idx, a.max_turns_per_episode))
                for i in idx:
                    prompt = context_window(messages[:i], a.history_turns)
                    f.write(json.dumps({"prompt": prompt, "completion": messages[i]["content"],
                                        "seed": seed, "difficulty": level}) + "\n")
                    rows += 1
    print(f"kept {kept}/{a.episodes * len(levels)} solved episodes, wrote {rows} rows to {a.out}")


if __name__ == "__main__":
    main()
