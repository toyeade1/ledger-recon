# Payment investigator

A local review console that proposes sandbox payment allocations, shows the source evidence, and records only the proposal a person approves. It extends this repository without changing the original RL environment, reward function, or benchmark.

## Run locally

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -e '.[investigator]'
ollama serve
# In another terminal; these are local model downloads, not API subscriptions.
ollama pull qwen2.5:7b
ollama pull nomic-embed-text
.venv/bin/python -m ledger_recon.investigator.server --mode ollama
```

Open http://127.0.0.1:8789 and select **Load demo**. Select a payment, investigate, inspect each quoted source, and approve the exact proposal. Records persist under `~/.local/share/ledger-investigator`. `--data /another/path` starts a separate company dataset. The listener binds only to loopback.

For the deterministic evidence baseline, pass `--mode rules`. The footer and every result identify the mode. An Ollama failure is a blocked investigation, not an invisible switch to rules. Dense retrieval may fall back to lexical retrieval; the result records the actual mode and reason.

## What happens

1. Import validated invoice/transaction records and supporting documents. SQLite is authoritative. IDs are immutable; a conflicting re-import fails atomically.
2. A persistent LangGraph workflow loads the payment, searches evidence through Qdrant, extracts a proposal, validates it, and saves the result.
3. Qdrant combines sparse token counts and local `nomic-embed-text` embeddings using reciprocal rank fusion. It is a candidate finder, not an accounting authority. Long document dense text is bounded; inspect the recorded retrieval note.
4. Local Qwen produces schema-constrained JSON. Independent code checks the selected records, exact quotations, supported evidence fields, customer, currency, cents and current balances. Unsupported or contradictory records stop the proposal.
5. Approval compares the exact immutable proposal, evidence revision and current balances inside a SQLite write transaction. Cash allocations sum to the payment. Credits reduce invoice balances and each credit identity can be consumed once. A second approval cannot allocate the payment twice.

LangGraph uses its real SQLite checkpointer. Qdrant runs embedded locally. The model and embedding provider are local Ollama. Three read-only MCP tools expose cases and evidence through the console API. Existing `ledger_recon.generator` supplies a seeded demo case; the discrepancy track and its expected answers remain separate from the original RL benchmark.

The design borrows the retrieval/evidence separation and approval principles from the author's doc-rag and agent-runtime projects. It does not require those services or the existing LLM gateway to be running, and does not claim their code was integrated as a dependency. CrewAI is not part of this implementation.

## Supported evidence

This first release handles one currency per payment case. Exact unique matches can be supported directly by invoice and transaction records. Partial/bundled payments need a remittance record. Credits need an explicit credit identity, invoice, customer, currency and integer-cent amount. It does not infer arbitrary commercial agreements from prose.

Import a JSON bundle:

```json
{
  "invoices": [{"id":"INV-1","customer":"Northstar","amount_cents":1000000,"currency":"USD"}],
  "transactions": [{"id":"PAY-1","customer":"Northstar","amount_cents":975000,"currency":"USD","reference":"November settlement"}],
  "documents": [
    {"id":"advice","filename":"advice.txt","text":"REMITTANCE transaction_id=PAY-1; customer=Northstar; currency=USD; allocations=INV-1:975000"},
    {"id":"credit","filename":"credit.txt","text":"Credit note: CN-1\nCustomer: Northstar\nCurrency: USD\nInvoice: INV-1\nCredit amount (cents): 25000"}
  ]
}
```

The alternative single-line credit format is:

```text
CREDIT credit_id=CN-1; customer=Northstar; currency=USD; invoice_id=INV-1; amount_cents=25000
```

Labelled remittance blocks use `Remittance advice`, `Payment:`, `Customer:`, `Currency:` and `Allocations (cents):`. Supporting TXT or text-based PDF files can be added through **Add evidence**. Image-only PDFs are rejected: OCR is not implemented. Every new imported record increments the evidence revision, so an outstanding proposal needs reinvestigation after evidence changes. The imported document is evidence for human review, not proof that an external credit is authentic.

Limits: 5 MB HTTP uploads, 50 PDF pages, 2,000 records of each kind per bundle, bounded source text. No bank connection, FX investigation, autonomous posting, ERP connector, multi-tenant service or production deployment. User approval writes only the local sandbox ledger.

## MCP

Run the console first, then configure a stdio MCP client with:

```json
{
  "command": "/absolute/path/to/ledger-recon/.venv/bin/python",
  "args": ["-m", "ledger_recon.investigator.mcp_server", "--url", "http://127.0.0.1:8789"]
}
```

Tools: `list_payment_cases`, `inspect_payment_case`, `search_payment_evidence`. The server has no approval or ledger-mutation tool. It queries the running local API, avoiding multiple processes opening the embedded Qdrant directory.

## Verification

Run the post-implementation functional workflow in `e2e/investigator_flow.py`. It uses temporary data and exercises complete import/investigate/review/approve workflows, including invalid imports, missing/conflicting evidence, stale proposals, credit replay, concurrency and restart. The separate evaluation report records synthetic/template coverage and local-model observations. It is not evidence of real customer savings or general accounting accuracy.
