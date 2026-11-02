"""Generate SFT warm-up data from the scripted solver.

The tic-tac-toe recipe in the talk: small models break the action format, so
before RL you do a short SFT pass to teach format and valid moves, then let RL
build the actual skill. Same idea here. Keep only fully solved episodes so you
don't bake in mistakes, and keep this small (a few hundred episodes): the goal
is format, not imitation of a script.

    python examples/make_sft_data.py --episodes 200 --out data/sft.jsonl
"""

import argparse
import json
from pathlib import Path

from ledger_recon import SYSTEM_PROMPT, LedgerReconEnv
from ledger_recon.agents import HeuristicAgent
from ledger_recon.protocol import render


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--difficulty", default="medium")
    p.add_argument("--out", type=Path, default=Path("data/sft.jsonl"))
    a = p.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    with a.out.open("w") as f:
        for seed in range(a.episodes):  # train seeds only (< 10000)
            env, agent = LedgerReconEnv(a.difficulty), HeuristicAgent(seed)
            obs, info = env.reset(seed=seed)
            messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": obs}]
            while True:
                act = agent.act(obs, info)
                messages.append({"role": "assistant", "content": render(act["name"], act.get("args"))})
                obs, reward, term, trunc, info = env.step(act)
                if term or trunc:
                    break
                messages.append({"role": "user", "content": obs})
            if info["reward_breakdown"]["solved"]:
                f.write(json.dumps({"messages": messages, "seed": seed}) + "\n")
                kept += 1
    print(f"wrote {kept}/{a.episodes} solved episodes to {a.out}")


if __name__ == "__main__":
    main()
