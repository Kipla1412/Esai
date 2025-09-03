
from src.scoring.bm25 import BM25

# 1. Config

config = {
    "terms": {"cachelimit": 1000000, "cutoff": 1.0},
    "content": True,
    "normalize": True,
    "k1": 1.5,
    "b": 0.75
}


# 2. Documents

docs = [
    ("doc1", "machine learning is amazing", None),
    ("doc2", "deep learning is a part of machine learning", None),
    ("doc3", "python is great for data science", None),
]

# 3. BM25 Setup

bm25 = BM25(config=config)

bm25.insert(docs)

# Build index
bm25.index()

# Debug: check stored documents
print("Indexed docs:", bm25.documents)

# 4. Query

query = "python machine learning"

# Search using BM25
results = bm25.search(query, limit=3)

print("\n🔍 BM25 Results:")
if results:
    for r in results:
        print(f"{r['id']} -> {r['score']:.4f}, text: {r['text']}")
else:
    print("No results found!")
