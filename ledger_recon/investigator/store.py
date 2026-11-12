"""SQLite is the accounting authority. Every mutation is transactional."""
from __future__ import annotations
from contextlib import contextmanager
import json
import sqlite3
from pathlib import Path
from .models import canonical, validate_bundle

SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value INTEGER NOT NULL);
INSERT OR IGNORE INTO metadata VALUES ('evidence_revision',0);
CREATE TABLE IF NOT EXISTS invoices (id TEXT PRIMARY KEY, data TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS transactions (id TEXT PRIMARY KEY, data TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS documents (id TEXT PRIMARY KEY, data TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS cases (id TEXT PRIMARY KEY REFERENCES transactions(id), status TEXT NOT NULL DEFAULT 'pending', proposal_id TEXT);
CREATE TABLE IF NOT EXISTS proposals (id TEXT PRIMARY KEY, case_id TEXT NOT NULL REFERENCES cases(id), revision INTEGER NOT NULL, data TEXT NOT NULL, created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')));
CREATE TABLE IF NOT EXISTS approvals (case_id TEXT PRIMARY KEY REFERENCES cases(id), proposal_id TEXT NOT NULL REFERENCES proposals(id), data TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS allocations (case_id TEXT NOT NULL REFERENCES approvals(case_id), invoice_id TEXT NOT NULL REFERENCES invoices(id), amount_cents INTEGER NOT NULL CHECK(amount_cents>0), PRIMARY KEY(case_id,invoice_id));
CREATE TABLE IF NOT EXISTS consumed_credits (identity TEXT PRIMARY KEY, credit_id TEXT NOT NULL, customer TEXT NOT NULL, invoice_id TEXT NOT NULL REFERENCES invoices(id), amount_cents INTEGER NOT NULL CHECK(amount_cents>0), document_id TEXT NOT NULL REFERENCES documents(id), case_id TEXT NOT NULL REFERENCES approvals(case_id));
CREATE TABLE IF NOT EXISTS events (sequence INTEGER PRIMARY KEY AUTOINCREMENT, case_id TEXT NOT NULL REFERENCES cases(id), stage TEXT NOT NULL, data TEXT NOT NULL, created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')));
"""


class Store:
    def __init__(self, root: Path):
        self.path = root / "ledger.sqlite3"
        with self.connect() as db:
            db.executescript(SCHEMA)

    @contextmanager
    def connect(self, write=False):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA busy_timeout=30000")
        try:
            if write:
                db.execute("BEGIN IMMEDIATE")
            yield db
            if write:
                db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    @staticmethod
    def revision(db):
        return db.execute("SELECT value FROM metadata WHERE key='evidence_revision'").fetchone()[0]

    @staticmethod
    def records(db, kind):
        if kind not in {"invoices", "transactions", "documents"}:
            raise ValueError("Invalid record kind")
        return [json.loads(row[0]) for row in db.execute(f"SELECT data FROM {kind} ORDER BY id")]

    @staticmethod
    def record(db, kind, record_id):
        if kind not in {"invoices", "transactions", "documents"}:
            raise ValueError("Invalid record kind")
        row = db.execute(f"SELECT data FROM {kind} WHERE id=?", (record_id,)).fetchone()
        if row is None:
            raise ValueError(f"Unknown {kind} ID: {record_id}")
        return json.loads(row[0])

    def import_bundle(self, bundle):
        bundle = validate_bundle(bundle)
        counts = {kind: 0 for kind in bundle}
        with self.connect(write=True) as db:
            for kind, records in bundle.items():
                for record in records:
                    data = canonical(record)
                    old = db.execute(f"SELECT data FROM {kind} WHERE id=?", (record["id"],)).fetchone()
                    if old and old[0] != data:
                        raise ValueError(f"Immutable {kind} ID conflicts with existing record: {record['id']}")
                    if not old:
                        db.execute(f"INSERT INTO {kind}(id,data) VALUES (?,?)", (record["id"], data))
                        counts[kind] += 1
                        if kind == "transactions":
                            db.execute("INSERT INTO cases(id) VALUES (?)", (record["id"],))
            total = sum(counts.values())
            if total:
                db.execute("UPDATE metadata SET value=value+? WHERE key='evidence_revision'", (total,))
            counts["evidence_revision"] = self.revision(db)
        return counts

    @staticmethod
    def event(db, case_id, stage, data):
        db.execute("INSERT INTO events(case_id,stage,data) VALUES (?,?,?)", (case_id, stage, canonical(data)))

    def log(self, case_id, stage, data):
        with self.connect(write=True) as db:
            self.event(db, case_id, stage, data)

    @staticmethod
    def balances(db):
        invoices = {r["id"]: r["amount_cents"] for r in Store.records(db, "invoices")}
        for table in ("allocations", "consumed_credits"):
            for row in db.execute(f"SELECT invoice_id,SUM(amount_cents) FROM {table} GROUP BY invoice_id"):
                invoices[row[0]] -= row[1]
        return invoices

    def snapshot(self, case_id, db=None):
        if db is None:
            with self.connect() as connection:
                connection.execute("BEGIN")
                return self.snapshot(case_id, connection)
        transaction = self.record(db, "transactions", case_id)
        return {"transaction": transaction, "invoices": self.records(db, "invoices"), "documents": self.records(db, "documents"), "balances": self.balances(db), "consumed_credits": [r[0] for r in db.execute("SELECT identity FROM consumed_credits")], "evidence_revision": self.revision(db)}

    def save_proposal(self, result):
        with self.connect(write=True) as db:
            current = db.execute("SELECT status FROM cases WHERE id=?", (result["case_id"],)).fetchone()
            if current[0] == "applied":
                return json.loads(db.execute("SELECT data FROM approvals WHERE case_id=?", (result["case_id"],)).fetchone()[0])
            db.execute("INSERT INTO proposals(id,case_id,revision,data) VALUES (?,?,?,?)", (result["proposal_id"], result["case_id"], result["evidence_revision"], canonical(result)))
            db.execute("UPDATE cases SET status=?,proposal_id=? WHERE id=?", (result["status"], result["proposal_id"], result["case_id"]))
            self.event(db, result["case_id"], "finalize", {"status": result["status"], "proposal_id": result["proposal_id"]})
        return result
