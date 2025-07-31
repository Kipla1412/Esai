
from src.vectors.dense import STVectors
import numpy as np
docs = [
    ("doc1", "The quick brown fox jumps over the lazy dog.", {}),
    ("doc2", "Sentence Transformers are powerful.", {}),
    ("doc3", "Vector search is useful for semantic retrieval.", {})
]

config = {
    "path": "all-MiniLM-L6-v2",      # You can use any SBERT model
    "encodebatch": 2,
    "dimensionality": 384,
    "gpu": False                     # Or True if you have GPU
}

vectorizer = STVectors(config=config, scoring=None, models={})

ids, dims, batches, stream = vectorizer.index(docs)

print("Vectorization complete")
print(f"Doc IDs: {ids}")
print(f"Dimension: {dims}")
print(f"Saved file: {stream}")


vectors = np.load(stream)
print("Vectors shape:", vectors.shape)
print(vectors[0][:10])

# from vectors.dense import STVectors

# config = {
#     "path": "all-MiniLM-L6-v2",
#     "encodebatch": 2,
#     "dimensionality": 384,
#     "gpu": False
# }

# vectorizer = STVectors(config=config, scoring=None, models={})

# # Encode new sentence
# text = "Graph neural networks are useful for social network analysis."
# vec = vectorizer.encode([text])

# print("Encoded Vector Shape:", vec.shape)
# print("Vector Sample:", vec[0][:10])  # print first 10 values
