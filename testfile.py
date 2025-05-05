"""
rag.py

A simple RAG (retrieval-augmented generation) pipeline using FAISS and a local embedding model.
This script allows the user to ask questions and retrieve news about the stock market from the past 24 hours.
"""

import json
import pickle
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
from faiss import IndexFlatL2

from embeddings import LocalEmbeddings     
from summarizer import summarize          

# ────────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ────────────────────────────────────────────────────────────────────────────────

JSONL_FILE    = "selenium_yahoo_finance.jsonl"
INDEX_FILE    = "faiss_index.pkl"
META_FILE     = "faiss_meta.pkl"
CUTOFF_HOURS  = 24
TOP_K         = 5

# ────────────────────────────────────────────────────────────────────────────────
# STEP 1: LOAD + OPTIONAL 24-H FILTER
# ────────────────────────────────────────────────────────────────────────────────

cutoff_dt = datetime.now(timezone.utc) - timedelta(hours=CUTOFF_HOURS)

raw = Path(JSONL_FILE).read_text(encoding="utf-8").splitlines()
docs = []
for line in raw:
    obj = json.loads(line)
    ts  = obj.get("timestamp")
    if ts:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if dt < cutoff_dt:
            continue
    docs.append({
        "id":        obj.get("id"),
        "title":     obj.get("title", "").strip(),
        "url":       obj.get("url", "").strip(),
        "timestamp": ts,
        "content":   obj.get("content_full") or obj.get("summary") or ""
    })

if not docs:
    print(f"No articles in the last {CUTOFF_HOURS} hours → nothing to index.", file=sys.stderr)
    sys.exit(1)

print(f"Loaded {len(docs)} articles (post-cutoff).", file=sys.stderr)


# ────────────────────────────────────────────────────────────────────────────────
# STEP 2: EMBEDDING + FAISS
# ────────────────────────────────────────────────────────────────────────────────

emb = LocalEmbeddings()
texts = [d["content"] for d in docs]
vectors = emb.embed_documents(texts)   # shape (n_docs, dim)

dim   = vectors.shape[1]
index = IndexFlatL2(dim)
index.add(np.asarray(vectors, dtype="float32"))

# persist index + metadata
with open(INDEX_FILE, "wb") as f_idx, open(META_FILE, "wb") as f_meta:
    pickle.dump(index, f_idx)
    pickle.dump(docs,  f_meta)

print(f"Index built and saved to {INDEX_FILE}, metadata to {META_FILE}.", file=sys.stderr)


# ────────────────────────────────────────────────────────────────────────────────
# STEP 3: QUERY LOOP
# ────────────────────────────────────────────────────────────────────────────────

def query_loop():
    # reload
    with open(INDEX_FILE, "rb") as f_idx, open(META_FILE, "rb") as f_meta:
        idx   = pickle.load(f_idx)
        meta  = pickle.load(f_meta)

    print("RAG ready — enter your query (empty to quit).", file=sys.stderr)
    while True:
        q = input("Query: ").strip()
        if not q:
            break

        qv, = emb.embed_documents([q])   # single vector
        D, I = idx.search(np.asarray([qv], dtype="float32"), TOP_K)

        for rank, doc_i in enumerate(I[0], start=1):
            doc = meta[doc_i]
            print(f"\n{rank}. {doc['title']}")
            print(f"   URL: {doc['url']}")
            summary = summarize(doc["content"])
            print(f"   Summary: {summary[:300].strip()}…")


if __name__ == "__main__":
    query_loop()
