
from src.scoring.tfidf import TFIDF

# Config

config = {
    "terms": {"cachelimit": 1000000, "cutoff": 0.5},  # <- important: set cutoff > 0
    "content": True,
    "normalize": True
}

docs = [
    ("doc1", "machine learning is amazing", None),
    ("doc2", "deep learning is a part of machine learning", None),
    ("doc3", "python is great for data science", None),
]
tfidf = TFIDF(config=config)

# Insert documents (convert to tokens)
tfidf.insert( docs)

# Build index
tfidf.index()

# Debug: check documents
print("Indexed docs:", tfidf.documents)


#  Query
query = "python "
#query_tokens = tfidf.tokenize(query.lower())

results = tfidf.search(query, limit=3)
#  Print results
print("\n🔍 TF-IDF Results:")
if results:
    for r in results:
        print(f"{r['id']} -> {r['score']:.4f}, text: {r['text']}")
else:
    print("No results found!")

