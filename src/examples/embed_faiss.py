from src.embeddings.base import Embeddings
from src.ann.dense.faiss import Faiss
import numpy as np

# Step 1: Embedding config
embed_config = {
    "method": "sentence-transformers",
    "path": "sentence-transformers/all-MiniLM-L6-v2",
    "gpu": True,
    "backend": "faiss",
    "sample": None,
    "mmap": False,
    "quantize": None
}

# Step 2: Sample documents
documents = [
    ("1", {"text": "Hello, this is the first sentence."}),
    ("2", {"text": "This is another example input for SBERT."}),
    ("3", {"text": "Sentence Transformers make semantic search easy."}),
]

# Step 3: Create embeddings + index in FAISS
with Embeddings(config=embed_config) as emb:
    ids, dimensions, embeddings = emb.index(documents)  # <-- we'll make it return these
    unique_ids, unique_indices = np.unique(ids, return_index=True)
    unique_embeddings = embeddings[unique_indices]

    # Add to FAISS
    emb.ann.index(unique_embeddings)

    # Save FAISS index
    emb.ann.save("docs.index")
    print("Index saved!")

faiss_loaded = Faiss(config =embed_config)
faiss_loaded.load("docs.index")
print("Index loaded!")

# Step 5: Run a query
query_docs = [("q1", {"text": "semantic search with transformers"})]
with Embeddings(config=embed_config) as emb:
    _, _, query_vecs = emb.index(query_docs)

results = faiss_loaded.search(query_vecs, limit=2)
print("Search results:", results)
