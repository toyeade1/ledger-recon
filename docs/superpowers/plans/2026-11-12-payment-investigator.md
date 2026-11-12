# Payment Investigator Implementation Plan

> For agentic workers: use superpowers:subagent-driven-development. User requires functional/E2E tests after implementation, overriding TDD. User approved design and execution; no further execution permission needed.

**Goal:** Deliver a local evidence-backed payment investigator and verified resume entry.
**Architecture:** Preserve existing RL environment. New investigator modules provide Qdrant retrieval, Ollama extraction, LangGraph checkpoints, authoritative SQLite proposals/approval and FastAPI UI/MCP access.
**Stack:** Python, FastAPI, LangGraph, Qdrant, Ollama, SQLite, MCP, pypdf.

## Chunk 1: Build the approved vertical slice

- [ ] Create isolated worktree outside iCloud, inspect existing source and preserve original benchmark. Write/review spec and plan.
- [ ] Implement ledger_recon/investigator/{__init__,models,store,retrieval,extraction,engine}.py following exact Engine contracts in spec. One implementer owns these modules. Add investigator optional dependencies and packaging entry points in pyproject.toml. Persistent SQLite/graph/Qdrant paths under explicit data root. Atomic immutable proposal approval and document-backed validation essential.
- [ ] Root implements ledger_recon/investigator/{server,mcp_server,demo}.py and static/index.html using Engine interface, with bounded upload/import and escaped UI. Demo uses seeded original generator for plain matching plus separately labelled evidence cases. No gold passed to Engine.
- [ ] Install optional dependencies in worktree .venv with uv. Start local Ollama using existing model; if absent/unavailable, inspect remote Alienware service, no paid API use. Choose an installed small model explicitly. Verify actual backend in results.
- [ ] After implementation write/run e2e/investigator_flow.py against temporary stores, real Qdrant, LangGraph and SQLite, HTTP API and browser. Validate wrong/stale/duplicate/concurrent approvals, crash recovery, import atomicity, malicious documents and blocked evidence. Exercise Ollama separately. No unit tests.
- [ ] Produce reproducible held-out synthetic evaluation in reports/payment-investigator.json and .md, compare exact matching baseline, distinguish rules/model modes, document data-generation limitations and errors.
- [ ] Request independent spec and code reviews; fix issues and rerun affected functional checks. Commit scoped changes, retain original source work untouched; existing remote branch only.
- [ ] Read current canonical master.json, back it up and update ledger_recon facts/bullet variants only with verified built capabilities/results. Preserve employer facts. Update ai-project-bank. Build an AI-engineering one-page resume preview in separate review folder, browser-free rendering, verify page count/text and visually inspect PNG. PDF/DOCX links in final.
- [ ] Checkpoint HQ implementation/results/evidence/next steps, inspect diff, save and verify remote commit. Report exact completed functionality and limitations.
