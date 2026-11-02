# ledger-recon

A multi-step RL environment where an LLM agent reconciles a multi-currency bank account against an invoice ledger, for a company that sells across Europe, the Middle East and Africa.

Most open RLVR data is math and code: one prompt, one answer, a checker. The model's answer doesn't change anything, and the next problem is just the next row. Real back-office work doesn't look like that. You look something up, act, and your earlier choices change what's left to do. This repo is a small, honest version of that kind of task.

```
13 transactions (page 1/2):
  TXN-44012 | 2026-09-09 |         -29.13 USD | SWIFT CHARGES
  TXN-34607 | 2026-09-19 |      65,686.93 AED | AL-NOOR TRADING INV 0738
  TXN-49274 | 2026-09-25 |      19,974.36 EUR | LYON BIOTECH SAS BULK PAYMENT 0393 0699
  TXN-87177 | 2026-09-28 |  13,655,373.54 NGN | LAGOS FINTECH PART PMT 1/2
  TXN-82850 | 2026-10-01 |         460.30 GBP | INTL TRF MOMBASA PORT INV-2026-0814 GBP
  TXN-11125 | 2026-10-08 |       3,761.52 USD | TRF FROM UNKNOWN REMITTER 88213
  ...
```

The agent has to work out that the GBP 460.30 pays a KES 76,199 invoice, that two Lagos transfers add up to one invoice, that one EUR transfer covers two Lyon invoices, and that the SWIFT charge and the unknown sender pay nothing.

## The task

Each episode is a procedurally generated set of books: invoices, a bank feed, and FX rates. Every transaction either pays one or more invoices or is noise. The agent decides which, using tools:

| Tool | What it does |
| --- | --- |
| `list_transactions(status, page)` | Bank feed, 8 rows per page |
| `list_invoices(status, page)` | Ledger, 8 rows per page; matched invoices drop off the open list |
| `search_invoices(customer, amount, currency)` | Fuzzy customer search and FX-aware amount search |
| `get_fx_rate(from, to)` | Scenario-specific rates (jittered per seed) |
| `match(transaction_id, invoice_ids)` | Allocate a payment to one or more invoices |
| `unmatch(transaction_id)` | Undo |
| `flag(transaction_id, reason)` | Mark as paying no invoice |
| `submit()` | End the episode |

Complications, mixed in by difficulty level:

| Kind | What's tricky | easy | medium | hard |
| --- | --- | --- | --- | --- |
| exact | Reference and amount line up (maybe minus a bank fee) | 4 | 4 | 4 |
| ref_typo | Reference mangled or missing; match on customer + amount | 2 | 2 | 3 |
| fx | Paid in a different currency than invoiced | 0 | 2 | 3 |
| split | One invoice paid in two transfers | 0 | 1 | 2 |
| bundle | One transfer pays two invoices | 0 | 1 | 2 |
| transposed | Reference points at a *different real invoice* | 0 | 0 | 2 |
| noise | Fees, interest, card refunds, unknown senders | 1 | 2 | 3 |
| decoy | Unpaid invoice with a near-identical amount, other customer | 0 | 2 | 3 |

Customer names show up the way bank feeds actually print them: `ALNOOR TRDG LLC`, `KRAKOW PREC SP ZOO`, `NBI FRESH PRODUCE`.

## Reward

For each transaction, compare the agent's label (a set of invoice ids, or `NO_MATCH`) with the gold label using Jaccard similarity. The episode score is the mean over transactions.

```
total = reconciliation + 0.1 * format - min(0.3, 0.02 * invalid_actions)
```

- Partial credit on bundles: one of two invoices scores 0.5.
- Over-matching hurts: pairing a transaction with five invoices when one is right scores 0.2.
- Flagging everything only earns credit on the real noise rows (about 0.15 reconciliation).
- Doing nothing scores 0.
- The env refuses to allocate more money to an invoice than it's owed, so you can't spray one invoice across every payment.
- Bad tool calls cost a little and the episode keeps going. Small models break format a lot early in training, and killing the episode on the first mistake starves them of signal.

## Baselines

50 held-out seeds per level (eval seeds start at 10000; train on anything below). None of these agents see the answers. They only use the tools.

```
agent        level    reward   recon  solved invalid/ep  turns
random       easy      0.203   0.126      0%       1.14   10.0
random       medium    0.151   0.083      0%       1.62   18.0
random       hard      0.098   0.058      0%       3.02   28.0
flag_all     easy      0.243   0.143      0%       0.00    9.0
flag_all     medium    0.254   0.154      0%       0.00   16.0
flag_all     hard      0.243   0.143      0%       0.00   25.0
amount_only  easy      1.083   0.983     94%       0.00   17.4
amount_only  medium    0.795   0.695      0%       0.00   41.4
amount_only  hard      0.697   0.597      0%       0.00   76.1
trust_ref    easy      1.000   0.900     44%       0.00   10.0
trust_ref    medium    0.805   0.705      0%       0.00   18.0
trust_ref    hard      0.677   0.577      0%       0.00   28.0
heuristic    easy      1.100   1.000    100%       0.00   10.0
heuristic    medium    1.100   1.000    100%       0.00   20.6
heuristic    hard      1.095   0.995     94%       0.00   33.2
```

How to read it:

- `amount_only` (match on amount, ignore names) is fine on easy and falls apart once decoys show up. `trust_ref` (believe the reference) dies on transposed references. Each complication breaks a specific shortcut, which is the point.
- `heuristic` is about 100 lines of hand-written domain logic. It proves every level is solvable through the tool interface alone. Its misses on hard are real ambiguity: a bundle total lands within fee tolerance of a third invoice, and only the narrative ("BULK PAYMENT 0617 0910", "PART PMT 2/2") tells you which is right. An agent that reads the text gets these; one that only does arithmetic doesn't.
- The interesting question isn't whether a script can solve it. It's whether a small model can learn this from reward alone, and whether that transfers to messier, real books.

I haven't run an LLM through it yet, so there are no model numbers here. That's the next commit.

## Quickstart

```bash
git clone https://github.com/toyeade1/ledger-recon && cd ledger-recon
pip install -e ".[dev,llm]"
pytest                                   # 24 tests: determinism, reward-hacking probes, solvability
python -m ledger_recon.eval              # baseline table above
```

Use it from code:

```python
from ledger_recon import LedgerReconEnv

env = LedgerReconEnv(difficulty="hard")
obs, info = env.reset(seed=42)
obs, reward, terminated, truncated, info = env.step(
    '<think>fee row</think><tool>{"name": "flag", "args": {"transaction_id": "TXN-44012", "reason": "bank fee"}}</tool>'
)
```

`step()` takes raw model text (parsed from `<tool>{json}</tool>`) or a `{"name", "args"}` dict. The return signature matches Gymnasium. Reward is 0 until the episode ends.

Run any OpenAI-compatible model (OpenAI, OpenRouter, vLLM, SGLang, Ollama):

```bash
python -m ledger_recon.eval --agent llm --model qwen3:4b \
  --base-url http://localhost:11434/v1 --difficulty easy --episodes 10 \
  --rollouts-out runs/qwen3-4b.jsonl
```

## Examples

- `examples/make_sft_data.py`: turns solved heuristic episodes into chat-format SFT data for a short format warm-up before RL.
- `examples/grpo_advantages.py`: GRPO's group-relative advantage on real rollouts, no GPU. It shows why every rollout in a group needs the same seed, and why a group where everyone succeeds teaches nothing.

## Design notes

- **Deterministic by seed.** GRPO compares rollouts from the same start. If the world differs between them, the advantage measures luck, not skill.
- **Train/eval split by seed range.** Easy to forget, easy to fool yourself.
- **Paginated, short observations.** Context length blows up fast in multi-step envs. Eight rows a page keeps a 90-turn episode tractable.
- **Fee tolerance is in the spec, not hidden.** The agent is told payments can be up to 1.5% short. A reward that depends on a rule the agent can't know is just noise.
- **Adversarial cases are designed, not accidental.** Each complication exists to break a shortcut a lazy policy would learn.

## Roadmap

- [ ] Eval a few open models (Qwen3, Gemma 3, LFM2) and a frontier model for a gap chart
- [ ] GRPO training run with TRL or prime-rl on easy, then a curriculum to hard
- [ ] Port to Prime Intellect's verifiers / Environments Hub and to OpenEnv
- [ ] Credit notes, write-offs, partial short-pays the agent must leave open
- [ ] A variant generated from anonymised real bank and ERP exports

## License

MIT
