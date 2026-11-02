"""What GRPO actually does with this environment's rewards, minus the GPU.

GRPO samples a *group* of G rollouts from the same starting state, scores each
one, and uses the group as its own baseline:

    advantage_i = (reward_i - mean(group)) / (std(group) + eps)

Positive advantage -> push up the probability of that trajectory's tokens.
Negative -> push down. No value network needed (that's what GRPO drops vs PPO).

Here the "policy" is the scripted solver with noise: with probability eps each
action is swapped for a random flag. Different eps = different skill. We show:
  1. advantages within a group sharing a seed (clean signal)
  2. what happens if you mix seeds into one group (the world's difficulty
     leaks into the advantage; this is why per-group seeding matters)
  3. a group where every rollout scores the same -> zero advantage, zero
     learning signal. Too-easy or too-hard tasks waste compute.

    python examples/grpo_advantages.py
"""

import random
import statistics

from ledger_recon import LedgerReconEnv, run_episode
from ledger_recon.agents import HeuristicAgent


class NoisyHeuristic(HeuristicAgent):
    def __init__(self, seed, eps):
        super().__init__(seed)
        self.eps = eps
        self.noise = random.Random(seed * 7919 + int(eps * 1000))

    def act(self, obs, info):
        a = super().act(obs, info)
        if a["name"] == "match" and self.noise.random() < self.eps:
            return {"name": "flag", "args": {"transaction_id": a["args"]["transaction_id"], "reason": "noise"}}
        return a


def advantages(rewards):
    mu, sd = statistics.mean(rewards), statistics.pstdev(rewards)
    return [(r - mu) / (sd + 1e-6) for r in rewards]


def show(title, rollouts):
    rewards = [r for _, r in rollouts]
    print(f"\n{title}\n  mean={statistics.mean(rewards):.3f} std={statistics.pstdev(rewards):.3f}")
    for (label, r), adv in zip(rollouts, advantages(rewards)):
        print(f"  {label:<26} reward={r:.3f}  advantage={adv:+.2f}")


def main():
    G, epsilons = 8, [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7]
    seed = 10_001

    group = []
    for k, eps in enumerate(epsilons[:G]):
        ro = run_episode(LedgerReconEnv("hard"), NoisyHeuristic(seed + 100 * k, eps), seed)
        group.append((f"seed={seed} eps={eps}", ro.reward))
    show("1) One group, same seed: advantage tracks skill", group)

    mixed = []
    for k, eps in enumerate([0.1] * G):
        s, lvl = seed + k, ["easy", "hard"][k % 2]
        ro = run_episode(LedgerReconEnv(lvl), NoisyHeuristic(s, eps), s)
        mixed.append((f"seed={s} {lvl} eps={eps}", ro.reward))
    show("2) Mixed worlds, identical skill: advantages still spread, and none of it is skill", mixed)

    flat = [(f"seed={seed} eps=0.0 #{k}", run_episode(LedgerReconEnv("easy"), HeuristicAgent(k), seed).reward)
            for k in range(4)]
    show("3) Everyone solves it: zero advantage, nothing to learn", flat)


if __name__ == "__main__":
    main()
