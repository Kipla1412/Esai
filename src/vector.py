from vectors.dense import STVectors

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

ids, dims, batches, path = vectorizer.index(docs)

print("✅ Vectorization complete")
print(f"Doc IDs: {ids}")
print(f"Dimension: {dims}")
print(f"Saved file: {path}")
