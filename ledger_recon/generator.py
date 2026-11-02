"""Procedural scenario generator for accounts-receivable reconciliation.

Every scenario is a function of (seed, difficulty). Same inputs, same books.
That matters for GRPO-style training: rollouts in a group must differ because
of the policy, not because the world changed underneath them.

A scenario has three parts:
  * invoices      - what customers owe (the ledger)
  * transactions  - what actually hit the bank account
  * ground truth  - which transactions pay which invoices, and which are noise

The agent never sees ground truth. It only sees invoices and transactions
through tools, so the reward can be checked exactly at the end.
"""

from __future__ import annotations

import random
import unicodedata
from dataclasses import dataclass, field
from datetime import date, timedelta

# Rough units of currency per 1 USD. Each scenario jitters these by up to 3%
# so a policy can't memorise one fixed table.
BASE_RATES_PER_USD: dict[str, float] = {
    "USD": 1.0,
    "EUR": 0.92,
    "GBP": 0.79,
    "NGN": 1550.0,
    "KES": 129.0,
    "AED": 3.6725,
    "SAR": 3.75,
    "EGP": 48.5,
    "ZAR": 18.2,
    "PLN": 3.95,
    "MAD": 9.9,
    "GHS": 15.4,
}


@dataclass(frozen=True)
class Company:
    name: str
    country: str
    currency: str
    # How the name tends to show up in bank narratives: truncated, uppercased,
    # transliterated. Real bank feeds look like this.
    bank_aliases: tuple[str, ...]


COMPANIES: tuple[Company, ...] = (
    Company("Adebayo Logistics Ltd", "NG", "NGN", ("ADEBAYO LOGISTICS", "ADEBAYO LOG LTD")),
    Company("Nairobi Fresh Produce Ltd", "KE", "KES", ("NAIROBI FRESH", "NBI FRESH PRODUCE")),
    Company("Al Noor Trading LLC", "AE", "AED", ("ALNOOR TRDG LLC", "AL-NOOR TRADING")),
    Company("Kraków Precision Sp. z o.o.", "PL", "PLN", ("KRAKOW PRECISION", "KRAKOW PREC SP ZOO")),
    Company("Nile Delta Textiles", "EG", "EGP", ("NILE DELTA TEXT", "NILEDELTA TEXTILES")),
    Company("Cape Solar (Pty) Ltd", "ZA", "ZAR", ("CAPE SOLAR PTY", "CAPESOLAR")),
    Company("Brightwater Analytics GmbH", "DE", "EUR", ("BRIGHTWATER ANALYTICS", "BRIGHTWATER GMBH")),
    Company("Thames Freight Ltd", "GB", "GBP", ("THAMES FRT LTD", "THAMES FREIGHT")),
    Company("Casablanca Agro SARL", "MA", "MAD", ("CASA AGRO SARL", "CASABLANCA AGRO")),
    Company("Accra Digital Services", "GH", "GHS", ("ACCRA DGTL SVCS", "ACCRA DIGITAL")),
    Company("Riyadh Medical Supplies Co", "SA", "SAR", ("RIYADH MED SUPPLIES", "RIYADH MEDICAL")),
    Company("Lyon Biotech SAS", "FR", "EUR", ("LYON BIOTECH", "LYON BIOTECH SAS")),
    Company("Lagos Fintech Hub Ltd", "NG", "NGN", ("LAGOS FINTECH", "LAGOS FINTECH HUB")),
    Company("Mombasa Port Services", "KE", "KES", ("MOMBASA PORT SVCS", "MOMBASA PORT")),
)

NO_MATCH = "NO_MATCH"


@dataclass(frozen=True)
class Invoice:
    invoice_id: str
    customer: str
    currency: str
    amount: float
    issue_date: str
    due_date: str


@dataclass(frozen=True)
class Transaction:
    transaction_id: str
    date: str
    description: str
    currency: str
    amount: float


@dataclass
class Scenario:
    seed: int
    difficulty: str
    invoices: list[Invoice]
    transactions: list[Transaction]
    rates_per_usd: dict[str, float]
    # transaction_id -> frozenset of invoice ids, or frozenset({NO_MATCH})
    gold: dict[str, frozenset[str]]
    # which complication produced each transaction; for analysis only
    kinds: dict[str, str] = field(default_factory=dict)

    def fx_rate(self, from_ccy: str, to_ccy: str) -> float:
        """Units of `to_ccy` per 1 unit of `from_ccy`."""
        return self.rates_per_usd[to_ccy] / self.rates_per_usd[from_ccy]


# How many of each complication appear at each level. "decoy" invoices are
# open invoices nobody paid; they look like plausible matches and should stay
# open. "noise" transactions (fees, refunds, strangers) should be flagged.
DIFFICULTY: dict[str, dict[str, int]] = {
    "easy": {"exact": 4, "ref_typo": 2, "fx": 0, "split": 0, "bundle": 0, "transposed": 0, "noise": 1, "decoy": 0},
    "medium": {"exact": 4, "ref_typo": 2, "fx": 2, "split": 1, "bundle": 1, "transposed": 0, "noise": 2, "decoy": 2},
    "hard": {"exact": 4, "ref_typo": 3, "fx": 3, "split": 2, "bundle": 2, "transposed": 2, "noise": 3, "decoy": 3},
}

# Bank/FX fees can shave up to this fraction off a payment. Tools tell the
# agent about this tolerance, so it is part of the task spec, not a secret.
MAX_FEE_FRACTION = 0.015


def ascii_upper(text: str) -> str:
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().upper()


def _round(amount: float) -> float:
    return round(amount + 1e-9, 2)


def generate(seed: int, difficulty: str = "medium") -> Scenario:
    if difficulty not in DIFFICULTY:
        raise ValueError(f"difficulty must be one of {sorted(DIFFICULTY)}")
    rng = random.Random(f"{seed}:{difficulty}")
    plan = DIFFICULTY[difficulty]

    rates = {c: r * (1 + rng.uniform(-0.03, 0.03)) for c, r in BASE_RATES_PER_USD.items()}
    rates["USD"] = 1.0

    start = date(2026, 9, 1)
    invoice_numbers = rng.sample(range(100, 999), 64)
    txn_numbers = rng.sample(range(10000, 99999), 64)
    invoices: list[Invoice] = []
    transactions: list[Transaction] = []
    gold: dict[str, frozenset[str]] = {}
    kinds: dict[str, str] = {}

    def new_invoice(company: Company, usd_value: float | None = None) -> Invoice:
        num = invoice_numbers.pop()
        issued = start + timedelta(days=rng.randint(0, 25))
        usd = usd_value if usd_value is not None else rng.uniform(400, 18000)
        # most customers are billed in local currency, some in EUR/USD
        currency = company.currency if rng.random() < 0.7 else rng.choice(["EUR", "USD"])
        inv = Invoice(
            invoice_id=f"INV-2026-{num:04d}",
            customer=company.name,
            currency=currency,
            amount=_round(usd * rates[currency]),
            issue_date=issued.isoformat(),
            due_date=(issued + timedelta(days=30)).isoformat(),
        )
        invoices.append(inv)
        return inv

    def new_txn(desc: str, currency: str, amount: float, after: str, kind: str, pays: frozenset[str]) -> None:
        num = txn_numbers.pop()
        when = date.fromisoformat(after) + timedelta(days=rng.randint(3, 40))
        tid = f"TXN-{num}"
        transactions.append(Transaction(tid, when.isoformat(), desc, currency, _round(amount)))
        gold[tid] = pays
        kinds[tid] = kind

    def alias(company: Company) -> str:
        return rng.choice(company.bank_aliases)

    def fee(amount: float) -> float:
        # most payments arrive whole; some lose a small bank charge
        return amount * (1 - rng.uniform(0, MAX_FEE_FRACTION)) if rng.random() < 0.5 else amount

    companies = list(COMPANIES)
    rng.shuffle(companies)

    def pick() -> Company:
        return rng.choice(companies)

    for _ in range(plan["exact"]):
        c = pick()
        inv = new_invoice(c)
        desc = f"TRF FROM {alias(c)} REF {inv.invoice_id}"
        new_txn(desc, inv.currency, fee(inv.amount), inv.issue_date, "exact", frozenset({inv.invoice_id}))

    for _ in range(plan["ref_typo"]):
        c = pick()
        inv = new_invoice(c)
        style = rng.choice(["nodash", "missing", "short"])
        if style == "nodash":
            ref = inv.invoice_id.replace("-", "")
            desc = f"{alias(c)} PAYMENT {ref}"
        elif style == "short":
            desc = f"{alias(c)} INV {inv.invoice_id[-4:]}"
        else:
            desc = f"PAYMENT {alias(c)} {date.fromisoformat(inv.issue_date).strftime('%b').upper()}"
        new_txn(desc, inv.currency, fee(inv.amount), inv.issue_date, "ref_typo", frozenset({inv.invoice_id}))

    for _ in range(plan["fx"]):
        c = pick()
        inv = new_invoice(c)
        pay_ccy = rng.choice([x for x in (c.currency, "EUR", "USD", "GBP") if x != inv.currency])
        paid = fee(inv.amount * rates[pay_ccy] / rates[inv.currency])
        desc = f"INTL TRF {alias(c)} {inv.invoice_id} {pay_ccy}"
        new_txn(desc, pay_ccy, paid, inv.issue_date, "fx", frozenset({inv.invoice_id}))

    for _ in range(plan["split"]):
        c = pick()
        inv = new_invoice(c)
        share = rng.uniform(0.3, 0.7)
        first = _round(inv.amount * share)
        second = _round(inv.amount - first)
        new_txn(f"{alias(c)} PART PMT 1/2", inv.currency, first, inv.issue_date, "split", frozenset({inv.invoice_id}))
        new_txn(f"{alias(c)} PART PMT 2/2", inv.currency, second, inv.issue_date, "split", frozenset({inv.invoice_id}))

    for _ in range(plan["bundle"]):
        c = pick()
        a, b = new_invoice(c), new_invoice(c)
        if b.currency != a.currency:
            b = Invoice(**{**b.__dict__, "currency": a.currency, "amount": _round(b.amount / rates[b.currency] * rates[a.currency])})
            invoices[-1] = b
        desc = f"{alias(c)} BULK PAYMENT {a.invoice_id[-4:]} {b.invoice_id[-4:]}"
        new_txn(desc, a.currency, a.amount + b.amount, b.issue_date, "bundle", frozenset({a.invoice_id, b.invoice_id}))

    for _ in range(plan["transposed"]):
        # The reference in the narrative points at a *different real invoice*.
        # Trusting the reference gives the wrong answer; checking the amount
        # and customer gives the right one.
        c = pick()
        target = new_invoice(c)
        other_company = rng.choice([x for x in companies if x.name != c.name])
        lure = new_invoice(other_company)  # stays unpaid: a decoy with a real id
        desc = f"TRF FROM {alias(c)} REF {lure.invoice_id}"
        new_txn(desc, target.currency, fee(target.amount), target.issue_date, "transposed", frozenset({target.invoice_id}))

    for _ in range(plan["decoy"]):
        # Unpaid invoice whose amount nearly equals a real payment from another
        # customer. Amount-only matching falls for it.
        if transactions:
            t = rng.choice(transactions)
            payer = {i.customer for i in invoices if i.invoice_id in gold[t.transaction_id]}
            c = rng.choice([x for x in companies if x.name not in payer])
            inv = Invoice(
                invoice_id=f"INV-2026-{invoice_numbers.pop():04d}",
                customer=c.name,
                currency=t.currency,
                amount=_round(t.amount * (1 + rng.uniform(-0.004, 0.004))),
                issue_date=(start + timedelta(days=rng.randint(0, 25))).isoformat(),
                due_date=(start + timedelta(days=rng.randint(30, 55))).isoformat(),
            )
            invoices.append(inv)
        else:
            new_invoice(pick())

    noise_templates = [
        ("MONTHLY ACCOUNT FEE", lambda: -rng.uniform(5, 40), "EUR"),
        ("SWIFT CHARGES", lambda: -rng.uniform(10, 35), "USD"),
        ("REFUND CARD 4471 TRAVEL", lambda: rng.uniform(80, 900), "EUR"),
        ("INTEREST CREDIT", lambda: rng.uniform(1, 60), "GBP"),
        ("TRF FROM UNKNOWN REMITTER 88213", lambda: rng.uniform(500, 5000), "USD"),
    ]
    for tpl in rng.sample(noise_templates, plan["noise"]):
        desc, amt, ccy = tpl
        new_txn(desc, ccy, amt(), start.isoformat(), "noise", frozenset({NO_MATCH}))

    rng.shuffle(invoices)
    transactions.sort(key=lambda t: (t.date, t.transaction_id))
    return Scenario(seed, difficulty, invoices, transactions, rates, gold, kinds)
