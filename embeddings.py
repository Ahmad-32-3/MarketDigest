from sentence_transformers import SentenceTransformer
import numpy as np

class LocalEmbeddings:
    """
    Wrapper around a SentenceTransformer model.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """
        Turn a list of strings into a (n_docs, dim) numpy array of embeddings.
        """
        # convert_to_numpy=True gives you a float32 np.ndarray out of the box
        return self.model.encode(texts, convert_to_numpy=True)
