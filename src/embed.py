from src.embeddings.base import Embeddings

config = {
    "dense": True
}

documents = [
    ("1", {"text": "Hello, this is the first sentence."}),
    ("2", {"text": "Another sample input goes here."}),
]

with Embeddings(config=config) as embedder:
    ids, dimensions, embeddings = embedder.index(documents)

    print("Embedding process completed.")
    print(f"Document IDs: {ids}")
    print(f"Embedding dimensions: {dimensions}")
    if embeddings is not None and len(embeddings) > 0:
        print(f"Sample embedding vector (first document):\n{embeddings[0]}")
    else:
        print("No embeddings generated.")