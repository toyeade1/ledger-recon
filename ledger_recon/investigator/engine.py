"""Persistent LangGraph investigation; explicit atomic human approval."""
from __future__ import annotations
import json
import re
from pathlib import Path
import sqlite3
import threading
from typing import TypedDict
from urllib.parse import urlparse
import uuid
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from .models import canonical
from .store import Store
from .retrieval import Retriever
from .extraction import derive, blocked, ollama_extract, validate_extracted, credit_identity


class Investigation(TypedDict, total=False):
    case_id: str
    snapshot: dict
    retrieved: list
    extracted: dict
    extraction_metadata: dict
    extraction_error: str
    retrieval_mode: str
    retrieval_note: str
    validated: dict
    result: dict


class Engine:
    def __init__(self, root: Path, mode="rules", model="qwen2.5:7b", ollama_url="http://127.0.0.1:11434", *, embeddings=True, interrupt_before=None):
        if mode not in {"rules", "ollama"}:
            raise ValueError("Extraction mode must be explicitly 'rules' or 'ollama'")
        parsed = urlparse(ollama_url)
        if parsed.scheme != "http" or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("Only a local HTTP Ollama endpoint is supported")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.mode, self.model, self.ollama_url = mode, model, ollama_url
        self.store = Store(self.root)
        self.retriever = Retriever(self.root, ollama_url, embeddings=embeddings)
        self._investigation_lock = threading.RLock()
        self._closed = False
        self._checkpoint_db = sqlite3.connect(self.root / "checkpoints.sqlite3", check_same_thread=False)
        self.checkpointer = SqliteSaver(self._checkpoint_db)
        graph = StateGraph(Investigation)
        for name in ("load", "retrieve", "extract", "validate", "finalize"):
            graph.add_node(name, getattr(self, "_" + name))
        graph.add_edge(START, "load")
        for left, right in zip(("load", "retrieve", "extract", "validate", "finalize"), ("retrieve", "extract", "validate", "finalize", END)):
            graph.add_edge(left, right)
        self.graph = graph.compile(checkpointer=self.checkpointer, interrupt_before=interrupt_before)

    def _ensure_open(self):
        if self._closed:
            raise RuntimeError("Engine is closed")

    def import_bundle(self, bundle):
        self._ensure_open()
        # Qdrant is derived state and is synchronized lazily from committed SQL.
        return self.store.import_bundle(bundle)

    def list_cases(self):
        self._ensure_open()
        with self.store.connect() as db:
            revision = self.store.revision(db)
            return [{**json.loads(row["data"]), "status": row["status"], "proposal_id": row["proposal_id"], "stale": row["revision"] is not None and row["revision"] != revision and row["status"] != "applied"} for row in db.execute("SELECT t.data,c.status,c.proposal_id,p.revision FROM cases c JOIN transactions t ON c.id=t.id LEFT JOIN proposals p ON c.proposal_id=p.id ORDER BY c.id")]

    def investigate(self, case_id):
        self._ensure_open()
        with self._investigation_lock:
            current = self.get_case(case_id)
            if current["status"] == "applied":
                return {k: current[k] for k in current if k not in {"transaction", "proposals", "events", "documents", "remaining_balances", "current_evidence_revision", "stale"}}
            # An unfinished run keeps its identity across restart. After finalization,
            # the new current proposal naturally starts a fresh immutable run.
            run_id = canonical([self.mode, self.model, case_id, current["current_evidence_revision"], current["proposal_id"]])
            config = {"configurable": {"thread_id": run_id}}
            checkpoint = self.graph.get_state(config)
            resume = bool(checkpoint.next)
            if resume:
                self.store.log(case_id, "resume", {"thread_id": run_id, "next": list(checkpoint.next)})
            outcome = self.graph.invoke(None if resume else {"case_id": case_id}, config=config)
            if "result" not in outcome:
                return {**self.get_case(case_id), "checkpoint_thread_id": run_id, "checkpoint_next": list(self.graph.get_state(config).next), "explanation": "Investigation checkpoint paused; investigate again to resume"}
            return outcome["result"]

    def _load(self, state):
        self.store.log(state["case_id"], "load", {"mode": self.mode})
        return {"snapshot": self.store.snapshot(state["case_id"])}

    def _retrieve(self, state):
        self.store.log(state["case_id"], "retrieve", {"status": "running"})
        tx = state["snapshot"]["transaction"]
        query = f"{tx['id']} {tx['customer']} {tx['currency']} {tx['amount_cents']} {tx['reference']}"
        try:
            retrieved = self.retriever.search(query[:4000], state["snapshot"]["documents"], limit=10)
            return {"retrieved": retrieved, "retrieval_mode": self.retriever.mode, "retrieval_note": self.retriever.note}
        except Exception as exc:
            return {"retrieved": [], "retrieval_mode": "retrieval-error", "retrieval_note": f"{type(exc).__name__}: {str(exc)[:500]}", "extraction_error": "Evidence retrieval failed: " + str(exc)[:500]}

    def _extract(self, state):
        self.store.log(state["case_id"], "extract", {"status": "running", "mode": self.mode})
        if state.get("extraction_error"):
            return {"extraction_metadata": {"provider": self.mode, "attempted": False}}
        if self.mode == "rules":
            result = derive(state["snapshot"])
            return {"extracted": {k: result[k] for k in ("allocations", "credits", "citations", "explanation", "blockers")}, "extraction_metadata": {"provider": "deterministic-rules", "model": None}}
        try:
            extracted, metadata = ollama_extract(state["snapshot"], state["retrieved"], self.model, self.ollama_url)
            return {"extracted": extracted, "extraction_metadata": metadata}
        except Exception as exc:
            reason = f"Local Ollama extraction failed ({type(exc).__name__}): {str(exc)[:500]}"
            return {"extraction_error": reason, "extraction_metadata": {"provider": "local-ollama", "model": self.model, "error": reason}}

    def _validate(self, state):
        self.store.log(state["case_id"], "validate", {"status": "running"})
        if state.get("extraction_error"):
            result = blocked(state["extraction_error"])
        else:
            result = validate_extracted(state["extracted"], state["snapshot"])
        return {"validated": result}

    def _finalize(self, state):
        result = {**state["validated"], "case_id": state["case_id"], "proposal_id": str(uuid.uuid4()), "status": "blocked" if state["validated"]["blockers"] else "review", "mode": self.mode, "retrieval_mode": state["retrieval_mode"], "retrieval_note": state["retrieval_note"], "extraction_metadata": state["extraction_metadata"], "evidence_revision": state["snapshot"]["evidence_revision"], "retrieved_document_ids": [r["document_id"] for r in state["retrieved"]]}
        return {"result": self.store.save_proposal(result)}

    def get_case(self, case_id):
        self._ensure_open()
        with self.store.connect() as db:
            db.execute("BEGIN")
            transaction = self.store.record(db, "transactions", case_id)
            case = db.execute("SELECT * FROM cases WHERE id=?", (case_id,)).fetchone()
            proposals = [{**json.loads(row["data"]), "created_at": row["created_at"]} for row in db.execute("SELECT data,created_at FROM proposals WHERE case_id=? ORDER BY rowid DESC", (case_id,))]
            approval = db.execute("SELECT data FROM approvals WHERE case_id=?", (case_id,)).fetchone()
            if approval:
                result = json.loads(approval[0])
            elif proposals:
                result = {k: v for k, v in proposals[0].items() if k != "created_at"}
            else:
                result = {"case_id": case_id, "status": "pending", "proposal_id": None, "allocations": [], "credits": [], "adjustment_cents": 0, "citations": [], "explanation": "Awaiting investigation", "blockers": [], "mode": self.mode, "retrieval_mode": "not-run", "evidence_revision": self.store.revision(db)}
            result["status"] = case["status"]
            result.update(transaction=transaction, proposals=proposals, events=[{"sequence": r["sequence"], "stage": r["stage"], "data": json.loads(r["data"]), "created_at": r["created_at"]} for r in db.execute("SELECT * FROM events WHERE case_id=? ORDER BY sequence", (case_id,))], documents=self.store.records(db, "documents"), remaining_balances=self.store.balances(db), current_evidence_revision=self.store.revision(db))
            result["stale"] = result["status"] != "applied" and bool(proposals) and result["evidence_revision"] != result["current_evidence_revision"]
            return result

    def approve(self, case_id, proposal_id):
        self._ensure_open()
        if not isinstance(proposal_id, str) or not proposal_id:
            raise ValueError("Approval requires the explicit current proposal_id")
        with self.store.connect(write=True) as db:
            case = db.execute("SELECT * FROM cases WHERE id=?", (case_id,)).fetchone()
            if case is None:
                raise ValueError("Unknown case")
            applied = db.execute("SELECT proposal_id,data FROM approvals WHERE case_id=?", (case_id,)).fetchone()
            if applied:
                if applied["proposal_id"] == proposal_id:
                    return json.loads(applied["data"])
                raise ValueError("Payment is already applied under a different proposal")
            if case["proposal_id"] != proposal_id:
                raise ValueError("Unknown or stale proposal: approve the current proposal explicitly")
            proposal = db.execute("SELECT * FROM proposals WHERE id=? AND case_id=?", (proposal_id, case_id)).fetchone()
            if proposal is None or case["status"] != "review":
                raise ValueError("Only a current review proposal can be approved; blocked cases cannot be applied")
            result = json.loads(proposal["data"])
            if proposal["revision"] != self.store.revision(db):
                raise ValueError("Stale evidence revision: reinvestigate after new records or documents")
            snapshot = self.store.snapshot(case_id, db)
            grounded = derive(snapshot)
            if grounded["blockers"]:
                raise ValueError("Current accounting validation failed: " + "; ".join(grounded["blockers"]))
            for key in ("allocations", "credits"):
                if sorted(map(canonical, grounded[key])) != sorted(map(canonical, result[key])):
                    raise ValueError("Proposal no longer matches current balances and supported financial evidence")
            if sum(a["amount_cents"] for a in result["allocations"]) != snapshot["transaction"]["amount_cents"]:
                raise ValueError("Allocations must sum to incoming transaction cents")
            customer = snapshot["transaction"]["customer"]
            result = {**result, "status": "applied", "approval": "explicit-human-intent"}
            db.execute("INSERT INTO approvals(case_id,proposal_id,data) VALUES (?,?,?)", (case_id, proposal_id, canonical(result)))
            for allocation in result["allocations"]:
                db.execute("INSERT INTO allocations(case_id,invoice_id,amount_cents) VALUES (?,?,?)", (case_id, allocation["invoice_id"], allocation["amount_cents"]))
            for credit in result["credits"]:
                db.execute("INSERT INTO consumed_credits(identity,credit_id,customer,invoice_id,amount_cents,document_id,case_id) VALUES (?,?,?,?,?,?,?)", (credit_identity(customer, credit["credit_id"]), credit["credit_id"], customer, credit["invoice_id"], credit["amount_cents"], credit["document_id"], case_id))
            if any(balance < 0 for balance in self.store.balances(db).values()):
                raise ValueError("Approval would overallocate an invoice")
            db.execute("UPDATE cases SET status='applied' WHERE id=?", (case_id,))
            self.store.event(db, case_id, "approve", {"proposal_id": proposal_id, "intent": "explicit", "allocations": result["allocations"], "credits": result["credits"]})
            return result

    def search(self, query, limit=5):
        self._ensure_open()
        if not isinstance(query, str) or not query.strip() or len(query) > 4000:
            raise ValueError("Query must contain 1–4000 characters")
        if type(limit) is not int or not 1 <= limit <= 50:
            raise ValueError("Search limit must be between 1 and 50")
        terms = set(re.findall(r"[\w.:-]+", query))
        authoritative = []
        with self.store.connect() as db:
            documents = self.store.records(db, "documents")
            for kind, prefix in (("invoices", "invoice:"), ("transactions", "transaction:")):
                for record in self.store.records(db, kind):
                    if record["id"] in terms or prefix + record["id"] in terms or str(record["amount_cents"]) in terms:
                        authoritative.append({"document_id": prefix + record["id"], "text": canonical(record), "score": 1.0, "retrieval_mode": "authoritative-sql"})
        retrieved = self.retriever.search(query, documents, limit)
        return (authoritative + retrieved)[:limit]

    def shutdown(self):
        if not self._closed:
            with self._investigation_lock:
                self.retriever.shutdown()
                self._checkpoint_db.close()
                self._closed = True
