# Payment investigator verification

Nine hand-authored templates; six unseen amount seeds. Synthetic functional coverage, not generalization or production accuracy.

Rules mode with real LangGraph, Qdrant sparse retrieval and SQLite: 54/54 expected case outcomes; 0 false actionable proposals. 30/30 evidence-supported proposals versus 12 from unique exact-amount matching. Median investigation time 0.0316 s. This baseline is deliberately simple and is not the original RL benchmark's stronger heuristic.

All six seeds repeat the same nine templates with varied amounts. These results do not estimate accuracy on unseen document formats, scanned invoices, real bank records or a different company. Embeddings and LLM extraction are disabled in this deterministic matrix.

Local model integration: qwen2.5:7b with qdrant-hybrid-rrf, one credit case, status review, 33.451 seconds, 4 validated citations. Earlier development model calls made cent/credit errors and were blocked; this successful-or-blocked smoke result is not an accuracy claim. Raw observed output is preserved in the JSON report. No paid model API used.

Post-implementation functional checks cover stale and superseded proposals, conflicting evidence, credit replay, concurrent approval, balance contention, persisted graph interruption/resume, invalid imports and untrusted document instructions. Browser/HTTP checks cover investigation, citations, approval, reload, PDF text extraction and blocked controls.
