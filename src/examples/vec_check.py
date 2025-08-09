from src.vectors.dense import STVectors
import numpy as np

docs = [
    ("doc1", "The quick brown fox jumps over the lazy dog.", {}),
    ("doc2", "Sentence Transformers are powerful.", {}),
    ("doc3", "Vector search is useful for semantic retrieval.", {}),
    ("doc4", "AI is transforming the world.", {}),
    ("doc5", "We are testing vector output.", {})
]

config = {
    "path": "all-MiniLM-L6-v2",
    "encodebatch": 2,
    "dimensionality": 384,
    "gpu": False
}

# Create vectorizer
vectorizer = STVectors(config=config, scoring=None, models={})

# Test vector generation
ids, dims, embs = vectorizer.vectors(
    documents=docs,
    batchsize=2,
    buffer="vectors_buffer.dat",  # memmap buffer
    dtype="float32"
)

# Output
print(" Vectorization complete!")
print(" Doc IDs:", ids)
print(" Dimensions:", dims)
print("Vector shape:", embs.shape)
print(" First 10 values of first vector:", embs[0][:10])
