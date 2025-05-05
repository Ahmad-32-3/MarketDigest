# rag.py

import pickle
from typing import List, Dict, Any, Tuple
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# ────────────────────────────────────────────────────────────
# CONFIGURATION
# ────────────────────────────────────────────────────────────

EMBED_MODEL     = "all-MiniLM-L6-v2"
QA_MODEL_NAME   = "google/flan-t5-base"
FAISS_INDEX_DIR = "faiss_index"
FAISS_META_FILE = "faiss_meta.pkl"

# ────────────────────────────────────────────────────────────
# 1) Load embeddings + index + metadata
# ────────────────────────────────────────────────────────────

embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

db = FAISS.load_local(
    FAISS_INDEX_DIR,
    embeddings,
    allow_dangerous_deserialization=True
)

with open(FAISS_META_FILE, "rb") as f:
    _METADATA = pickle.load(f)

_retriever = db.as_retriever(search_kwargs={"k": 5})

# ────────────────────────────────────────────────────────────
# 2) Lazy‐load FLAN-T5 pipeline on first use
# ────────────────────────────────────────────────────────────

_qa_pipe = None

def _get_qa_pipe():
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
    tokenizer = AutoTokenizer.from_pretrained(QA_MODEL_NAME)
    model     = AutoModelForSeq2SeqLM.from_pretrained(QA_MODEL_NAME)
    return pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto"
    )

# ────────────────────────────────────────────────────────────
# 3) Public function
# ────────────────────────────────────────────────────────────

def answer_query(query: str) -> Tuple[str, List[Dict[str, Any]]]:
    """
    1) retrieve top-k articles
    2) build prompt from titles+content
    3) generate answer with FLAN-T5
    """
    global _qa_pipe
    if _qa_pipe is None:
        _qa_pipe = _get_qa_pipe()

    docs = _retriever.get_relevant_documents(query)
    context = "\n\n".join(f"{i+1}. {d.metadata['title']}\n{d.page_content}"
                          for i, d in enumerate(docs))

    prompt = (
        "Answer ONLY from these snippets:\n\n"
        f"{context}\n\nQuestion: {query}\nAnswer:"
    )

    out = _qa_pipe(prompt, max_length=200, truncation=True, do_sample=False)
    answer = out[0]["generated_text"].strip()
    sources = [d.metadata for d in docs]
    return answer, sources
