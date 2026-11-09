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

### First open model: Qwen2.5-7B-Instruct

20 held-out seeds per level, temperature 0.7, Q4_K_M GGUF on llama.cpp (RTX 4070 laptop, 8 GB). The agent keeps the system prompt, the first observation and the last 12 exchanges.

```
agent        level    reward   recon  solved invalid/ep  turns
qwen2.5-7b   easy     -0.104   0.143      0%      31.95   33.5
qwen2.5-7b   medium   -0.194   0.065      0%      52.25   59.2
qwen2.5-7b   hard     -0.205   0.047      0%      78.90   85.1
```

It scores below random on every level, and almost all of that is format, not reasoning. Only 32 of 671 turns on easy were valid tool calls. The usual failure is writing a reasonable call like `search_invoices(customer=None, amount=50554.34, currency="PLN")` with the JSON but without the `<tool>` tags, then repeating that same line until the turn budget runs out. 48 of the 60 episodes got stuck repeating one action five or more times. The sliding window makes it worse: after a few bad turns, the only examples left in context are its own mistakes.

When it does produce valid calls, it sometimes gets them right, including bundles. So the next step is the one from the tic-tac-toe talk: a short SFT pass to teach the format, then RL for the skill. Pass `--rollouts-out` to save every transcript, and read a few before trusting any average.

### SFT format warm-up: Qwen2.5-1.5B-Instruct, before and after

LoRA (rank 16, all linear layers) on 1,500 turns from solved heuristic episodes on train seeds, one epoch, about an hour on the same 8 GB laptop GPU (5.4 GB peak). Each training example is cut with the exact context window the agent sees at eval time. Both evals use transformers in bf16, temperature 0.7, a 40-turn window and the same 10 held-out seeds per level. The wider window is deliberate: the scripted solver reads the invoice list once and matches against it 20+ turns later, so with 12 turns the training targets would reference invoices the model can no longer see.

```
model             level    reward   recon  solved invalid/ep  turns
1.5b base         easy     -0.265   0.014      0%      34.40   35.5
1.5b + SFT        easy      0.911   0.843     50%       2.50   14.5
1.5b base         medium   -0.297   0.000      0%      61.80   62.0
1.5b + SFT        medium    0.709   0.715     10%      11.10   34.0
1.5b base         hard     -0.299   0.000      0%      91.70   92.1
1.5b + SFT        hard      0.346   0.518      0%      47.00   78.5
```

The base model almost never produces a valid call and runs out of turns in 28 of 30 episodes. After SFT it follows the format and the basic routine: list, check FX, match or flag, submit. Per case type on hard, it does well on FX payments (0.80) and plain matches (0.68), so-so on bundles (0.57), and badly on split payments (0.25) and transposed references (0.10). In other words, it learned the procedure and still falls for the traps that are there to break shortcuts.

Long episodes are where it falls apart. On hard, 270 of its errors are trying to match a transaction it already matched, and 94 use transaction ids that don't exist. Half the hard episodes hit the turn budget.

That's about what SFT should do: fix format and give RL a policy that gets non-zero reward. Separating "looks right" from "is right" on transposed refs and splits is the job for GRPO. Ten seeds per level is a small sample, so read the gaps as rough. These numbers aren't comparable to the 7B table above (different window and runtime).

Reproduce:

```bash
pip install torch transformers peft
python examples/make_sft_data.py --format turns --difficulty all --episodes 150 \
  --max-turns-per-episode 12 --history-turns 40 --out data/sft_turns.jsonl
python examples/sft_lora.py --data data/sft_turns.jsonl --out runs/sft-1.5b --max-rows 1500
python -m ledger_recon.eval --agent llm --backend hf --model Qwen/Qwen2.5-1.5B-Instruct \
  --adapter runs/sft-1.5b --history-turns 40 --max-tokens 256 --episodes 10
```

On an 8 GB card the transformers backend is slow for the base model, because its long rambling turns push memory past the card and Windows spills into shared RAM. For bigger evals, merge the adapter (`examples/merge_lora.py`), convert to GGUF and serve both models through llama.cpp.

## Quickstart

```bash
git clone https://github.com/toyeade1/ledger-recon && cd ledger-recon
pip install -e ".[dev,llm]"
pytest                                   # 25 tests: determinism, reward-hacking probes, solvability
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
- `examples/sft_lora.py`: LoRA SFT in plain PyTorch, sized for an 8 GB GPU. Loss on the action tokens only.
- `examples/merge_lora.py`: merge an adapter into the base model for serving.
- `examples/grpo_advantages.py`: GRPO's group-relative advantage on real rollouts, no GPU. It shows why every rollout in a group needs the same seed, and why a group where everyone succeeds teaches nothing.

## Design notes

- **Deterministic by seed.** GRPO compares rollouts from the same start. If the world differs between them, the advantage measures luck, not skill.
- **Train/eval split by seed range.** Easy to forget, easy to fool yourself.
- **Paginated, short observations.** Context length blows up fast in multi-step envs. Eight rows a page keeps a 90-turn episode tractable.
- **Fee tolerance is in the spec, not hidden.** The agent is told payments can be up to 1.5% short. A reward that depends on a rule the agent can't know is just noise.
- **Adversarial cases are designed, not accidental.** Each complication exists to break a shortcut a lazy policy would learn.

## Roadmap

- [x] First open-model eval (Qwen2.5-7B-Instruct, above)
- [x] SFT format warm-up on a small model, then eval before and after (above)
- [ ] Eval more open models (Qwen3, Gemma 3, LFM2) and a frontier model for a gap chart
- [ ] GRPO from the SFT checkpoint on easy, then a curriculum to hard
- [ ] Port to Prime Intellect's verifiers / Environments Hub and to OpenEnv
- [ ] Credit notes, write-offs, partial short-pays the agent must leave open
- [ ] A variant generated from anonymised real bank and ERP exports

## License

MIT
