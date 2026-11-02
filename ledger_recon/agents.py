"""Baseline policies.

None of these see ground truth. They only use the tool interface, same as an
LLM would, which is what makes them useful:

  RandomAgent        floor. What does flailing score?
  FlagAllAgent       reward-hacking probe: flag everything and submit.
  AmountOnlyAgent    naive solver: match on amount alone. Falls for decoys.
  TrustRefAgent      naive solver: trust the reference in the narrative. Falls for transposed refs.
  HeuristicAgent     careful scripted solver. Shows the task is solvable through the tools,
                     and gives a reference trajectory you could use for SFT warm-up.

Agents receive (observation_text, info) and return an action: either raw text
for the env to parse, or a {"name", "args"} dict.
"""

from __future__ import annotations

import itertools
import random
import re
from typing import Any, Generator

from .env import _name_match
from .generator import MAX_FEE_FRACTION

Action = dict[str, Any]
NOISE_WORDS = ("FEE", "CHARGES", "INTEREST", "REFUND", "UNKNOWN")
STOPWORDS = {"TRF", "FROM", "REF", "PAYMENT", "PART", "PMT", "BULK", "INTL", "INV", "LTD", "LLC", "SAS",
             "SARL", "GMBH", "PTY", "SVCS", "SERVICES", "THE", "AND", "SEP", "OCT", "AUG",
             "EUR", "USD", "GBP", "NGN", "KES", "AED", "SAR", "EGP", "ZAR", "PLN", "MAD", "GHS"}
REF_RE = re.compile(r"INV-?2026-?(\d{4})|INV (\d{4})")


class ScriptedAgent:
    """Agents written as generators: `data = yield action` receives info['data'] back."""

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)
        self._gen: Generator[Action, Any, None] | None = None

    def plan(self) -> Generator[Action, Any, None]:  # pragma: no cover - abstract
        raise NotImplementedError

    def act(self, obs: str, info: dict) -> Action:
        if self._gen is None:
            self._gen = self.plan()
            return next(self._gen)
        try:
            return self._gen.send(info.get("data"))
        except StopIteration:
            return {"name": "submit", "args": {}}

    # shared helper: page through a listing tool
    def _collect(self, tool: str, status: str) -> Generator[Action, Any, list[dict]]:
        rows: list[dict] = []
        page = 1
        while True:
            chunk = yield {"name": tool, "args": {"status": status, "page": page}}
            if not chunk:
                return rows
            rows.extend(chunk)
            if len(chunk) < 8:
                return rows
            page += 1


class RandomAgent(ScriptedAgent):
    def plan(self):
        txns = yield from self._collect("list_transactions", "all")
        invs = yield from self._collect("list_invoices", "all")
        for t in txns:
            if self.rng.random() < 0.3:
                yield {"name": "flag", "args": {"transaction_id": t["transaction_id"], "reason": "random"}}
            else:
                pick = self.rng.choice(invs)["invoice_id"]
                yield {"name": "match", "args": {"transaction_id": t["transaction_id"], "invoice_ids": [pick]}}
        yield {"name": "submit", "args": {}}


class FlagAllAgent(ScriptedAgent):
    def plan(self):
        txns = yield from self._collect("list_transactions", "all")
        for t in txns:
            yield {"name": "flag", "args": {"transaction_id": t["transaction_id"], "reason": "lazy"}}
        yield {"name": "submit", "args": {}}


class _FxMixin:
    def _rate(self, frm: str, to: str):
        if frm == to:
            return 1.0
        key = (frm, to)
        if key not in self._fx:
            self._fx[key] = yield {"name": "get_fx_rate", "args": {"from_currency": frm, "to_currency": to}}
        return self._fx[key]

    def _amount_ok(self, txn_amt: float, rate: float, inv_amt: float) -> bool:
        conv = abs(txn_amt) * rate
        return inv_amt * (1 - MAX_FEE_FRACTION) - 0.01 <= conv <= inv_amt * 1.005 + 0.01


class AmountOnlyAgent(ScriptedAgent, _FxMixin):
    def plan(self):
        self._fx: dict = {}
        txns = yield from self._collect("list_transactions", "all")
        invs = yield from self._collect("list_invoices", "all")
        used: set[str] = set()
        for t in txns:
            hit = None
            for inv in invs:
                if inv["invoice_id"] in used:
                    continue
                rate = yield from self._rate(t["currency"], inv["currency"])
                if self._amount_ok(t["amount"], rate, inv["amount"]):
                    hit = inv["invoice_id"]
                    break
            if hit:
                used.add(hit)
                yield {"name": "match", "args": {"transaction_id": t["transaction_id"], "invoice_ids": [hit]}}
            else:
                yield {"name": "flag", "args": {"transaction_id": t["transaction_id"], "reason": "no amount match"}}
        yield {"name": "submit", "args": {}}


class TrustRefAgent(ScriptedAgent):
    def plan(self):
        txns = yield from self._collect("list_transactions", "all")
        invs = yield from self._collect("list_invoices", "all")
        by_suffix = {i["invoice_id"][-4:]: i["invoice_id"] for i in invs}
        for t in txns:
            m = REF_RE.search(t["description"])
            if m:
                hit = by_suffix.get(m.group(1) or m.group(2))
                if hit:
                    yield {"name": "match", "args": {"transaction_id": t["transaction_id"], "invoice_ids": [hit]}}
                    continue
            yield {"name": "flag", "args": {"transaction_id": t["transaction_id"], "reason": "no ref"}}
        yield {"name": "submit", "args": {}}


class HeuristicAgent(ScriptedAgent, _FxMixin):
    """Customer first, then reference, then amount. Handles fx, splits and bundles."""

    def plan(self):
        self._fx: dict = {}
        txns = yield from self._collect("list_transactions", "all")
        invs = yield from self._collect("list_invoices", "all")
        done_txn: set[str] = set()
        used_inv: set[str] = set()

        def customer_invoices(desc: str) -> list[dict]:
            words = [w for w in re.findall(r"[A-Z][A-Z\-]{2,}", desc) if w not in STOPWORDS]
            return [i for i in invs if i["invoice_id"] not in used_inv
                    and any(_name_match(w, i["customer"]) for w in words)]

        for t in txns:
            tid = t["transaction_id"]
            if tid in done_txn:
                continue
            desc = t["description"].upper()
            if t["amount"] <= 0 or any(w in desc for w in NOISE_WORDS):
                done_txn.add(tid)
                yield {"name": "flag", "args": {"transaction_id": tid, "reason": "fee/interest/refund/unknown"}}
                continue

            cands = customer_invoices(desc)
            m = REF_RE.search(desc)
            suffix = (m.group(1) or m.group(2)) if m else None
            if suffix:
                cands.sort(key=lambda i: i["invoice_id"][-4:] != suffix)  # referenced invoice first

            chosen: list[str] | None = None
            for inv in cands:  # single invoice
                rate = yield from self._rate(t["currency"], inv["currency"])
                if self._amount_ok(t["amount"], rate, inv["amount"]):
                    chosen = [inv["invoice_id"]]
                    break
            if chosen is None:  # bundle: two invoices, same customer
                for a, b in itertools.combinations(cands, 2):
                    if a["currency"] != b["currency"]:
                        continue
                    rate = yield from self._rate(t["currency"], a["currency"])
                    if self._amount_ok(t["amount"], rate, a["amount"] + b["amount"]):
                        chosen = [a["invoice_id"], b["invoice_id"]]
                        break
            if chosen:
                done_txn.add(tid)
                used_inv.update(chosen)
                yield {"name": "match", "args": {"transaction_id": tid, "invoice_ids": chosen}}
                continue

            # split: this txn + another unresolved txn from the same customer sum to one invoice
            partner = None
            for other in txns:
                oid = other["transaction_id"]
                if oid == tid or oid in done_txn or other["currency"] != t["currency"]:
                    continue
                shared = {i["invoice_id"] for i in customer_invoices(other["description"].upper())}
                for inv in cands:
                    if inv["invoice_id"] not in shared:
                        continue
                    rate = yield from self._rate(t["currency"], inv["currency"])
                    if self._amount_ok(t["amount"] + other["amount"], rate, inv["amount"]):
                        partner = (oid, inv["invoice_id"])
                        break
                if partner:
                    break
            if partner:
                oid, iid = partner
                done_txn.update({tid, oid})
                used_inv.add(iid)
                yield {"name": "match", "args": {"transaction_id": tid, "invoice_ids": [iid]}}
                yield {"name": "match", "args": {"transaction_id": oid, "invoice_ids": [iid]}}
                continue

            done_txn.add(tid)
            yield {"name": "flag", "args": {"transaction_id": tid, "reason": "no matching invoice"}}
        yield {"name": "submit", "args": {}}


AGENTS = {
    "random": RandomAgent,
    "flag_all": FlagAllAgent,
    "amount_only": AmountOnlyAgent,
    "trust_ref": TrustRefAgent,
    "heuristic": HeuristicAgent,
}
