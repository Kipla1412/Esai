from src.embeddings.base import Embeddings

config = {
    "dense": True,               # enable transformer embeddings
    "scoring": "bm25",           # enable BM25 sparse scoring
    "indexes": {                 # multi-index setup
        "products": {"dense": True},      # dense search on products
        "reviews": {"scoring": "tfidf"}   # sparse search on reviews
    }
}

emb = Embeddings(config)

documents =[
    (1, "iPhone 14 Pro Max has the best camera and battery life", None),
    (2, "Samsung Galaxy S23 Ultra is amazing for photography", None),
    (3, "Google Pixel 7 Pro offers excellent AI features", None),
    (4, "Best budget phone is OnePlus Nord with fast charging", None),
]

emb.index(documents)

query ="best phone for camera"
results = emb.search(query, limit =3)

print("/n search Results:")

print(results)