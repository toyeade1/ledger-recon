"""Run one episode and keep the whole trajectory.

Inspecting rollouts by eye is the single most useful debugging habit in RL
work: aggregate reward tells you *that* something is off, the transcript tells
you *what*.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol

from .env import LedgerReconEnv


class Agent(Protocol):
    def act(self, obs: str, info: dict) -> str | dict: ...


@dataclass
class Rollout:
    seed: int
    difficulty: str
    agent: str
    steps: list[dict[str, Any]] = field(default_factory=list)
    reward: float = 0.0
    breakdown: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(self.__dict__, default=str)


def run_episode(env: LedgerReconEnv, agent: Agent, seed: int, agent_name: str = "agent") -> Rollout:
    obs, info = env.reset(seed=seed)
    ro = Rollout(seed, env.difficulty, agent_name)
    ro.steps.append({"role": "env", "obs": obs})
    while True:
        action = agent.act(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        ro.steps.append({"role": "agent", "action": action})
        ro.steps.append({"role": "env", "obs": obs})
        if terminated or truncated:
            ro.reward = reward
            ro.breakdown = info["reward_breakdown"]
            return ro
