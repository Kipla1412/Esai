
from src.embeddings.base import Embeddings
from src.ann.dense.faiss import Faiss
import numpy as np

embed_config ={
    "method ": "sentence-transformers",
    "path": "sentence-transformers/all-MiniLM-L6-v2",
    "gpu" : True,
    "backend":"faiss",
    "sample": None,
    "mmap":False,
    "quantize":None
}

documents = [
    ("1", {"text": "Hello, this is the first sentence."}),
    ("2", {"text": "This is another example input for SBERT."}),
    ("3", {"text": "Sentence Transformers make semantic search easy."}),
]

with Embeddings(config = embed_config) as emb:

    ids,dimensions,embeddings =emb.index(documents)

    unique_ids,unique_indices = np.unique(ids,return_index=True)
    unique_embeddings = embeddings[unique_indices]

    emb.ann.index(unique_embeddings)
    emb.ann.save("doc.index")
    print("index sucessfully")

loaded_docs = Faiss(config=embed_config)
loaded_docs.load("doc.index")

print ("loaded sucessfully")

query_docs = [("q1", {"text": "semantic search with transformers"})]
with Embeddings(config = embed_config) as emb:
    _,_,vectors =emb.index(query_docs)

results = loaded_docs.search(vectors, limit=2)
print("search results:",results)

