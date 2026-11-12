"""Local Qdrant lexical retrieval with optional Ollama dense rank fusion."""
from __future__ import annotations
from collections import Counter
import hashlib
import re
import threading
import uuid
import httpx
from qdrant_client import QdrantClient, models


def lexical(text):
    counts = Counter(re.findall(r"[\w.-]+", text.lower()))
    hashed = Counter()
    for token, count in counts.items():
        hashed[int.from_bytes(hashlib.sha256(token.encode()).digest()[:4], "big")] += count
    return models.SparseVector(indices=sorted(hashed), values=[float(hashed[i]) for i in sorted(hashed)])


class Retriever:
    def __init__(self, root, ollama_url, embeddings=True):
        self.client = QdrantClient(path=str(root / "qdrant"))
        self.http = httpx.Client(base_url=ollama_url, timeout=60, trust_env=False)
        self.embeddings = embeddings
        self.lock = threading.RLock()
        self.mode = "qdrant-lexical-only"
        self.note = "Dense retrieval has not been requested"
        if not self.client.collection_exists("evidence_lexical"):
            self.client.create_collection("evidence_lexical", vectors_config={}, sparse_vectors_config={"lexical": models.SparseVectorParams()})

    @staticmethod
    def point_id(document_id):
        return str(uuid.uuid5(uuid.NAMESPACE_URL, "ledger-recon-evidence:" + document_id))

    def indexed_ids(self, collection):
        result = set()
        offset = None
        while True:
            points, offset = self.client.scroll(collection, limit=256, offset=offset, with_payload=["document_id"], with_vectors=False)
            result.update(point.payload["document_id"] for point in points)
            if offset is None:
                return result

    def sync(self, documents):
        existing = self.indexed_ids("evidence_lexical")
        pending = [d for d in documents if d["id"] not in existing]
        for start in range(0, len(pending), 64):
            self.client.upsert("evidence_lexical", points=[models.PointStruct(id=self.point_id(d["id"]), vector={"lexical": lexical(d["text"])}, payload={"document_id": d["id"], "text": d["text"], "filename": d["filename"]}) for d in pending[start:start+64]])

    def embed(self, texts):
        response = self.http.post("/api/embed", json={"model": "nomic-embed-text", "input": texts})
        response.raise_for_status()
        embeddings = response.json()["embeddings"]
        if len(embeddings) != len(texts) or not embeddings or not embeddings[0]:
            raise ValueError("Ollama returned invalid embedding shape")
        return embeddings

    def search(self, query, documents, limit=5):
        if not isinstance(query, str) or not query.strip() or len(query) > 4000:
            raise ValueError("Query must contain 1–4000 characters")
        if type(limit) is not int or not 1 <= limit <= 50:
            raise ValueError("Search limit must be between 1 and 50")
        with self.lock:
            self.sync(documents)
            self.mode = "qdrant-lexical-only"
            self.note = "Dense embeddings disabled" if not self.embeddings else "No documents to embed"
            sparse = lexical(query)
            matches = self.client.query_points("evidence_lexical", query=sparse, using="lexical", limit=max(limit, 20)).points if sparse.indices and documents else []
            rankings = [matches]
            if self.embeddings and documents:
                try:
                    tags = self.http.get("/api/tags", timeout=2)
                    tags.raise_for_status()
                    names = [m["name"].split(":")[0] for m in tags.json().get("models", [])]
                    if "nomic-embed-text" not in names:
                        raise ValueError("nomic-embed-text is not installed in local Ollama")
                    vector = self.embed([query])[0]
                    collection = "evidence_nomic"
                    if not self.client.collection_exists(collection):
                        self.client.create_collection(collection, vectors_config=models.VectorParams(size=len(vector), distance=models.Distance.COSINE))
                    existing = self.indexed_ids(collection)
                    pending = [d for d in documents if d["id"] not in existing]
                    for start in range(0, len(pending), 16):
                        batch = pending[start:start+16]
                        vectors = self.embed([d["text"][:6000] for d in batch])
                        self.client.upsert(collection, points=[models.PointStruct(id=self.point_id(d["id"]), vector=v, payload={"document_id": d["id"], "text": d["text"], "filename": d["filename"]}) for d, v in zip(batch, vectors)])
                    rankings.append(self.client.query_points(collection, query=vector, limit=max(limit, 20)).points)
                    self.mode = "qdrant-hybrid-rrf"
                    self.note = "Qdrant sparse lexical + local nomic-embed-text dense reciprocal-rank fusion; dense text bounded to first 6000 characters"
                except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
                    self.note = f"Lexical-only fallback: {type(exc).__name__}: {str(exc)[:240]}"
            scores, payloads = Counter(), {}
            for ranking in rankings:
                for rank, point in enumerate(ranking, 1):
                    identifier = point.payload["document_id"]
                    scores[identifier] += 1.0 / (60 + rank)
                    payloads[identifier] = point.payload
            return [{"document_id": identifier, "text": payloads[identifier]["text"], "score": score, "retrieval_mode": self.mode} for identifier, score in scores.most_common(limit)]

    def shutdown(self):
        self.http.close()
        self.client.close()
