import json
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def main():
    docs = []
    # 1) Read your JSONL
    with open("selenium_yahoo_finance.jsonl", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            docs.append(Document(
                page_content   = obj.get("content_full", ""),
                metadata       = {
                    "id":        obj.get("id"),
                    "title":     obj.get("title"),
                    "url":       obj.get("url"),
                    "timestamp": obj.get("timestamp"),
                    "ticker":    obj.get("ticker"),
                },
            ))

    # 2) Embed locally
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_documents(docs, embeddings)

    # 3) Persist your FAISS index
    db.save_local("faiss_index")
    print(f"Indexed {len(docs)} docs into faiss_index")

if __name__ == "__main__":
    main()
