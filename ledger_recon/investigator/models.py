"""Bounded import contracts and canonical ledger serialization."""
from __future__ import annotations
import json
import re

ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,95}$")
CURRENCY = re.compile(r"^[A-Z]{3}$")
MAX_CENTS = 10**12


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def positive_cents(value):
    return type(value) is int and 0 < value <= MAX_CENTS


def valid_id(value):
    return isinstance(value, str) and bool(ID.fullmatch(value))


def validate_bundle(bundle):
    if not isinstance(bundle, dict) or set(bundle) - {"invoices", "transactions", "documents"}:
        raise ValueError("Bundle must contain only invoices, transactions, and documents")
    result = {}
    for kind in ("invoices", "transactions", "documents"):
        records = bundle.get(kind, [])
        if not isinstance(records, list) or len(records) > 2000:
            raise ValueError(f"{kind} must be a list of at most 2000 records")
        clean = {}
        for record in records:
            if not isinstance(record, dict) or not valid_id(record.get("id")):
                raise ValueError(f"Invalid {kind} record ID")
            if kind == "documents":
                allowed = {"id", "text", "filename"}
                if record["id"].startswith(("invoice:", "transaction:")):
                    raise ValueError("Document IDs cannot use reserved invoice: or transaction: citation prefixes")
                if not isinstance(record.get("text"), str) or not record["text"].strip() or len(record["text"]) > 200000:
                    raise ValueError("Document text must contain 1–200000 characters")
                normalized = {"id": record["id"], "text": record["text"], "filename": record.get("filename", record["id"] + ".txt")}
                if not isinstance(normalized["filename"], str) or len(normalized["filename"]) > 255:
                    raise ValueError("Invalid document filename")
            else:
                allowed = {"id", "customer", "amount_cents", "currency"}
                if kind == "transactions":
                    allowed.add("reference")
                if not isinstance(record.get("customer"), str) or not 1 <= len(record["customer"]) <= 160 or record["customer"] != record["customer"].strip() or any(c in record["customer"] for c in "\n\r;"):
                    raise ValueError("Customer must be a nonempty bounded single-line name")
                if not positive_cents(record.get("amount_cents")):
                    raise ValueError("Amounts must be positive integer cents (booleans are invalid)")
                if not isinstance(record.get("currency"), str) or not CURRENCY.fullmatch(record["currency"]):
                    raise ValueError("Currency must be a three-letter uppercase code")
                normalized = {k: record[k] for k in ("id", "customer", "amount_cents", "currency")}
                if kind == "transactions":
                    normalized["reference"] = record.get("reference", "")
                    if not isinstance(normalized["reference"], str) or len(normalized["reference"]) > 1000:
                        raise ValueError("Invalid transaction reference")
            if set(record) - allowed:
                raise ValueError(f"Unsupported fields in {kind}")
            if record["id"] in clean and clean[record["id"]] != normalized:
                raise ValueError(f"Conflicting duplicate ID {record['id']}")
            clean[record["id"]] = normalized
        result[kind] = list(clean.values())
    if sum(len(d["text"]) for d in result["documents"]) > 5000000:
        raise ValueError("Bundle document text exceeds 5 million characters")
    return result
