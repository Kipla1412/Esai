from src.embeddings.base import Embeddings
from src.pipeline.llm.rag import RAG

embed_config = {
    "method": "sentence-transformers",
    "path": "sentence-transformers/all-MiniLM-L6-v2",
    "gpu": True,
    "backend": "faiss"
}

# ---- First batch ----
docs = [
    ("1", {"text": "Hello, this is the first sentence."}),
    ("2", {"text": "This is another example input for SBERT."}),
]

emb = Embeddings(config=embed_config)
emb.index(docs)   # ✅ no reindex, just normal index

# ---- Second batch ----
more_docs = [
    ("3", {"text": "Sentence Transformers make semantic search easy."}),
    ("4", {"text": "RAG combines retrieval and generation."}),
]

emb.index(more_docs)   # ✅ adds to same FAISS index, doesn’t wipe old ones

# ---- Use in RAG ----
rag = RAG(
    similarity=emb,
    path="gpt2",
    gpu=True,
    context=2,
    template="Answer based only on context:\n{context}\n\nQ: {question}\nA:",
)

answers = rag(["What is RAG?"])
print(answers)
