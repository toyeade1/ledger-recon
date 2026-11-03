"""LedgerReconEnv: a multi-step, tool-using RL environment for LLM agents.

The agent plays an accounts-receivable clerk at a company that sells across
Europe, the Middle East and Africa. It has a bank feed and an invoice ledger
and must decide, for every bank transaction, which invoice(s) it pays or
whether it is noise (fees, refunds, unknown senders).

Unlike a static Q&A dataset, actions change the state:
  * matching a transaction removes it (and its invoices) from the open lists
  * a wrong match keeps the right invoice hidden until the agent unmatches it
  * every lookup costs a step out of a fixed budget

API mirrors Gymnasium without depending on it:

    env = LedgerReconEnv(difficulty="medium")
    obs, info = env.reset(seed=7)
    obs, reward, terminated, truncated, info = env.step(model_text_or_dict)

Reward is 0 on every step except the last, where it is the full verifiable
score from rewards.score(). info["reward_breakdown"] has the components.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from . import protocol
from .generator import MAX_FEE_FRACTION, NO_MATCH, Invoice, Scenario, Transaction, ascii_upper, generate
from .rewards import RewardBreakdown, score

PAGE_SIZE = 8

TOOL_DOCS = """\
Tools (call exactly one per turn):
  list_transactions(status="unresolved"|"all", page=1)   bank feed, 8 per page
  list_invoices(status="open"|"all", page=1)             ledger, 8 per page
  search_invoices(customer=None, amount=None, currency=None)
        customer: any word from the name (case/accents ignored)
        amount: finds invoices within 1.5% of this amount after converting to each invoice's currency
        currency: currency the amount is given in (required when amount is set)
  get_fx_rate(from_currency, to_currency)                units of to_currency per 1 from_currency
  match(transaction_id, invoice_ids)                     invoice_ids is a list; use 2+ ids for a bulk payment
  unmatch(transaction_id)                                undo a match or flag
  flag(transaction_id, reason)                           transaction pays no invoice (fee, refund, unknown sender)
  submit()                                               finish; unresolved transactions score 0"""

SYSTEM_PROMPT = f"""\
You are an accounts-receivable clerk reconciling a multi-currency bank account.
For each bank transaction decide which open invoice(s) it pays, or flag it if it pays none.

Rules of thumb:
  - Payments can arrive up to 1.5% short because of bank charges.
  - A payment can be in a different currency than its invoice. Convert with get_fx_rate.
  - One invoice can be paid in two parts, and one payment can cover two invoices.
  - References in bank narratives can be mangled or point at the wrong invoice. Check customer and amount too.
  - Fees, interest, card refunds and unknown senders pay no invoice: flag them.

{TOOL_DOCS}

Reply format: optional reasoning in <think>...</think>, then exactly one tool call:
<tool>{{"name": "<tool>", "args": {{...}}}}</tool>"""


@dataclass
class EpisodeState:
    scenario: Scenario
    matches: dict[str, frozenset[str]] = field(default_factory=dict)
    flags: dict[str, str] = field(default_factory=dict)
    turns: int = 0
    valid_turns: int = 0
    invalid_actions: int = 0
    done: bool = False


class ToolError(Exception):
    pass


class LedgerReconEnv:
    def __init__(self, difficulty: str = "medium", max_turns: int | None = None):
        self.difficulty = difficulty
        self.max_turns = max_turns
        self.state: EpisodeState | None = None

    # ------------------------------------------------------------------ API
    def reset(self, seed: int = 0) -> tuple[str, dict]:
        scenario = generate(seed, self.difficulty)
        self.state = EpisodeState(scenario)
        n_txn = len(scenario.transactions)
        # budget scales with problem size: room to look things up, not to brute force
        self._budget = self.max_turns or 4 * n_txn + 10
        obs = (
            f"Reconcile the bank account for September-October 2026.\n"
            f"{n_txn} bank transactions, {len(scenario.invoices)} invoices in the ledger. "
            f"You have {self._budget} turns.\n\n"
            + self._list_transactions("unresolved", 1)[0]
        )
        return obs, {"seed": seed, "difficulty": self.difficulty, "turn_budget": self._budget}

    def step(self, action: str | dict) -> tuple[str, float, bool, bool, dict]:
        st = self._require_state()
        if st.done:
            raise RuntimeError("Episode is over. Call reset().")
        st.turns += 1

        if isinstance(action, str):
            parsed = protocol.parse(action)
        else:
            parsed = protocol.ParsedAction(action.get("name"), action.get("args", {}) or {})

        data: Any = None
        if not parsed.ok:
            st.invalid_actions += 1
            obs = f"ERROR: {parsed.error}"
        else:
            st.valid_turns += 1
            try:
                obs, data = self._dispatch(parsed.name, parsed.args)
            except ToolError as e:
                st.invalid_actions += 1
                obs = f"ERROR: {e}"
            except (TypeError, ValueError, AttributeError, KeyError) as e:
                # wrong, missing or badly typed arguments, e.g. amount="" or page="two".
                # Model output must never crash the env; it is just an invalid turn.
                st.invalid_actions += 1
                obs = f"ERROR: bad arguments for {parsed.name}: {e}"

        terminated = st.done
        truncated = False
        if not terminated and st.turns >= self._budget:
            truncated = True
            st.done = True
            obs += "\n\nTurn budget used up. Submitting what you have."

        info: dict[str, Any] = {"data": data, "turn": st.turns, "turns_left": self._budget - st.turns}
        reward = 0.0
        if st.done:
            breakdown = self.breakdown()
            reward = breakdown.total
            info["reward_breakdown"] = breakdown.as_dict()
        else:
            obs += f"\n[{self._budget - st.turns} turns left, {self._unresolved_count()} transactions unresolved]"
        return obs, reward, terminated, truncated, info

    def breakdown(self) -> RewardBreakdown:
        st = self._require_state()
        return score(st.scenario, st.matches, set(st.flags), st.valid_turns, st.turns, st.invalid_actions)

    # --------------------------------------------------------------- tools
    def _dispatch(self, name: str | None, args: dict) -> tuple[str, Any]:
        tools = {
            "list_transactions": self._list_transactions,
            "list_invoices": self._list_invoices,
            "search_invoices": self._search_invoices,
            "get_fx_rate": self._get_fx_rate,
            "match": self._match,
            "unmatch": self._unmatch,
            "flag": self._flag,
            "submit": self._submit,
        }
        if name not in tools:
            raise ToolError(f"Unknown tool {name!r}. Available: {', '.join(tools)}.")
        out = tools[name](**args)
        return out if isinstance(out, tuple) else (out, None)

    def _list_transactions(self, status: str = "unresolved", page: int = 1):
        st = self._require_state()
        if status not in ("unresolved", "all"):
            raise ToolError('status must be "unresolved" or "all".')
        rows = [t for t in st.scenario.transactions if status == "all" or not self._resolved(t.transaction_id)]
        chunk, header = self._page(rows, page, "transactions")
        lines = [header] + [self._fmt_txn(t) for t in chunk]
        return "\n".join(lines), [t.__dict__ for t in chunk]

    def _list_invoices(self, status: str = "open", page: int = 1):
        st = self._require_state()
        if status not in ("open", "all"):
            raise ToolError('status must be "open" or "all".')
        allocated = self._allocated_invoices()
        rows = [i for i in st.scenario.invoices if status == "all" or i.invoice_id not in allocated]
        chunk, header = self._page(rows, page, "invoices")
        lines = [header] + [self._fmt_inv(i, i.invoice_id in allocated) for i in chunk]
        return "\n".join(lines), [i.__dict__ for i in chunk]

    def _search_invoices(self, customer: str | None = None, amount: float | None = None, currency: str | None = None):
        st = self._require_state()
        if customer is None and amount is None:
            raise ToolError("Give customer, amount, or both.")
        if amount is not None:
            if currency is None or currency not in st.scenario.rates_per_usd:
                raise ToolError("currency is required with amount and must be a known ISO code.")
            amount = float(amount)
        allocated = self._allocated_invoices()
        hits: list[Invoice] = []
        for inv in st.scenario.invoices:
            if customer is not None and not _name_match(customer, inv.customer):
                continue
            if amount is not None:
                converted = abs(amount) * st.scenario.fx_rate(currency, inv.currency)
                if not (inv.amount * (1 - MAX_FEE_FRACTION) - 0.01 <= converted <= inv.amount * 1.005 + 0.01):
                    continue
            hits.append(inv)
        if not hits:
            return "No invoices match.", []
        hits = hits[:PAGE_SIZE]
        lines = [f"{len(hits)} match(es):"] + [self._fmt_inv(i, i.invoice_id in allocated) for i in hits]
        return "\n".join(lines), [i.__dict__ for i in hits]

    def _get_fx_rate(self, from_currency: str, to_currency: str):
        rates = self._require_state().scenario.rates_per_usd
        for c in (from_currency, to_currency):
            if c not in rates:
                raise ToolError(f"Unknown currency {c!r}. Known: {', '.join(sorted(rates))}.")
        r = self._require_state().scenario.fx_rate(from_currency, to_currency)
        return f"1 {from_currency} = {r:.6f} {to_currency}", r

    def _match(self, transaction_id: str, invoice_ids: list[str] | str):
        st = self._require_state()
        txn = self._txn(transaction_id)
        if isinstance(invoice_ids, str):
            invoice_ids = [invoice_ids]
        if not invoice_ids:
            raise ToolError("invoice_ids is empty.")
        known = {i.invoice_id for i in st.scenario.invoices}
        unknown = [i for i in invoice_ids if i not in known]
        if unknown:
            raise ToolError(f"Unknown invoice id(s): {', '.join(unknown)}.")
        if self._resolved(txn.transaction_id):
            raise ToolError(f"{txn.transaction_id} is already resolved. unmatch it first.")
        # An invoice may receive several partial payments, so we don't block
        # re-use. We do refuse obvious over-allocation: that would let a policy
        # spray one invoice across every transaction.
        for iid in invoice_ids:
            inv = self._inv(iid)
            already = sum(
                self._to(t.amount, t.currency, inv.currency) / len(ids)
                for tid, ids in st.matches.items() if iid in ids for t in [self._txn(tid)]
            )
            incoming = self._to(txn.amount, txn.currency, inv.currency) / len(invoice_ids)
            if already > 0 and already + incoming > inv.amount * 1.01:
                raise ToolError(f"{iid} is already fully paid by earlier matches.")
        st.matches[txn.transaction_id] = frozenset(invoice_ids)
        return f"Matched {txn.transaction_id} -> {', '.join(invoice_ids)}."

    def _unmatch(self, transaction_id: str):
        st = self._require_state()
        txn = self._txn(transaction_id)
        if st.matches.pop(txn.transaction_id, None) is None and st.flags.pop(txn.transaction_id, None) is None:
            raise ToolError(f"{txn.transaction_id} was not matched or flagged.")
        return f"{txn.transaction_id} is unresolved again."

    def _flag(self, transaction_id: str, reason: str = ""):
        st = self._require_state()
        txn = self._txn(transaction_id)
        if self._resolved(txn.transaction_id):
            raise ToolError(f"{txn.transaction_id} is already resolved. unmatch it first.")
        st.flags[txn.transaction_id] = str(reason)[:200]
        return f"Flagged {txn.transaction_id} as not paying any invoice."

    def _submit(self):
        st = self._require_state()
        st.done = True
        return f"Submitted. {self._unresolved_count()} transactions were left unresolved."

    # ------------------------------------------------------------- helpers
    def _require_state(self) -> EpisodeState:
        if self.state is None:
            raise RuntimeError("Call reset() first.")
        return self.state

    def _txn(self, tid: str) -> Transaction:
        for t in self._require_state().scenario.transactions:
            if t.transaction_id == tid:
                return t
        raise ToolError(f"Unknown transaction id {tid!r}.")

    def _inv(self, iid: str) -> Invoice:
        for i in self._require_state().scenario.invoices:
            if i.invoice_id == iid:
                return i
        raise ToolError(f"Unknown invoice id {iid!r}.")

    def _to(self, amount: float, frm: str, to: str) -> float:
        return amount * self._require_state().scenario.fx_rate(frm, to)

    def _resolved(self, tid: str) -> bool:
        st = self._require_state()
        return tid in st.matches or tid in st.flags

    def _unresolved_count(self) -> int:
        return sum(not self._resolved(t.transaction_id) for t in self._require_state().scenario.transactions)

    def _allocated_invoices(self) -> set[str]:
        return {iid for ids in self._require_state().matches.values() for iid in ids}

    @staticmethod
    def _page(rows: list, page: int, noun: str):
        page = max(1, int(page))
        pages = max(1, -(-len(rows) // PAGE_SIZE))
        chunk = rows[(page - 1) * PAGE_SIZE: page * PAGE_SIZE]
        return chunk, f"{len(rows)} {noun} (page {min(page, pages)}/{pages}):"

    def _fmt_txn(self, t: Transaction) -> str:
        st = self._require_state()
        tag = ""
        if t.transaction_id in st.matches:
            tag = f"  [matched -> {', '.join(sorted(st.matches[t.transaction_id]))}]"
        elif t.transaction_id in st.flags:
            tag = "  [flagged]"
        return f"  {t.transaction_id} | {t.date} | {t.amount:>14,.2f} {t.currency} | {t.description}{tag}"

    @staticmethod
    def _fmt_inv(i: Invoice, allocated: bool) -> str:
        tag = "  [allocated]" if allocated else ""
        return f"  {i.invoice_id} | {i.customer} | {i.amount:>14,.2f} {i.currency} | issued {i.issue_date} due {i.due_date}{tag}"


def _name_match(query: str, name: str) -> bool:
    compact = "".join(ch for ch in ascii_upper(name) if ch.isalnum())
    tokens = ["".join(ch for ch in tok if ch.isalnum()) for tok in ascii_upper(query).split()]
    tokens = [t for t in tokens if len(t) >= 3]
    return bool(tokens) and any(t in compact for t in tokens)


__all__ = ["LedgerReconEnv", "SYSTEM_PROMPT", "TOOL_DOCS", "NO_MATCH"]
