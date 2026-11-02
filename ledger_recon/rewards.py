"""Verifiable reward for a finished reconciliation.

Each bank transaction has one gold label: the set of invoices it pays, or
{NO_MATCH} for noise. The agent's label for that transaction is whatever it
matched or flagged; anything it never touched gets an empty set.

Score per transaction = Jaccard(gold, predicted). Episode score = the mean.

Why Jaccard per transaction instead of accuracy:
  * partial credit on bundles (got 1 of 2 invoices -> 0.5, not 0)
  * over-matching is punished (matching a txn to 5 invoices when 1 is right -> 0.2)
  * flagging everything only earns credit on the real noise rows
  * doing nothing earns 0

Shaping terms are kept small next to the task score, so the cheapest way to
raise reward is still to reconcile correctly.
"""

from __future__ import annotations

from dataclasses import dataclass

from .generator import NO_MATCH, Scenario

FORMAT_WEIGHT = 0.1
INVALID_PENALTY = 0.02
MAX_INVALID_PENALTY = 0.3


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


@dataclass
class RewardBreakdown:
    reconciliation: float     # mean per-transaction Jaccard, in [0, 1]
    format: float             # share of turns that parsed into a valid tool call
    invalid_penalty: float    # >= 0, subtracted
    solved: bool              # every transaction exactly right (for pass@k)
    per_kind: dict[str, float]

    @property
    def total(self) -> float:
        return self.reconciliation + FORMAT_WEIGHT * self.format - self.invalid_penalty

    def as_dict(self) -> dict:
        return {
            "total": round(self.total, 4),
            "reconciliation": round(self.reconciliation, 4),
            "format": round(self.format, 4),
            "invalid_penalty": round(self.invalid_penalty, 4),
            "solved": self.solved,
            "per_kind": {k: round(v, 4) for k, v in self.per_kind.items()},
        }


def score(
    scenario: Scenario,
    matches: dict[str, frozenset[str]],
    flags: set[str],
    valid_turns: int,
    total_turns: int,
    invalid_actions: int,
) -> RewardBreakdown:
    per_txn: dict[str, float] = {}
    for tid, gold in scenario.gold.items():
        if tid in flags:
            pred = frozenset({NO_MATCH})
        else:
            pred = matches.get(tid, frozenset())
        per_txn[tid] = jaccard(gold, pred) if pred else 0.0

    recon = sum(per_txn.values()) / len(per_txn)

    by_kind: dict[str, list[float]] = {}
    for tid, s in per_txn.items():
        by_kind.setdefault(scenario.kinds.get(tid, "?"), []).append(s)

    return RewardBreakdown(
        reconciliation=recon,
        format=(valid_turns / total_turns) if total_turns else 0.0,
        invalid_penalty=min(MAX_INVALID_PENALTY, INVALID_PENALTY * invalid_actions),
        solved=all(s == 1.0 for s in per_txn.values()),
        per_kind={k: sum(v) / len(v) for k, v in sorted(by_kind.items())},
    )
