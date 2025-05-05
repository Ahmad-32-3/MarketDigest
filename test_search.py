from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# 1) Reconstruct the same embeddings object you used for ingestion
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 2) Load your saved index — opt into de-serialization since it’s trusted
db = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

# 3) Do a similarity search
query = "Fed decision on interest rates"
results = db.similarity_search(query, k=5)

print("Top 5 matches for:", query)
for doc in results:
    print("-", doc.metadata["title"])
