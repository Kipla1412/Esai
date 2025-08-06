from src.embeddings.base import Embeddings
import numpy as np

config = {
    "method": "sentence-transformers",
    "path": "sentence-transformers/all-MiniLM-L6-v2",  # or another SBERT model id / local path
    "gpu": True
}
documents = [
    ("1", {"text": "Hello, this is the first sentence."}),
    ("2", {"text": "This is another example input for SBERT."}),
    ("3", {"text": "Sentence Transformers make semantic search easy."}),
]

with Embeddings(config=config) as emb:
    ids, dimensions, embeddings = emb.index(documents)
    unique_ids, unique_indices = np.unique(ids, return_index=True)
    unique_embeddings = embeddings[unique_indices]

    print("Dimensions:", dimensions)
    print("IDs:", ids)
    print("Embeddings shape:", embeddings.shape)
    
    print("\n--- Vectors ---")
    for uid, vec in zip(unique_ids, unique_embeddings):
        print(f"ID: {uid} -> First 10 values: {vec[:10].tolist()}")
