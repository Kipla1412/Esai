
from src.ann.dense.faiss import Faiss
import numpy as np

config = {

    "backend": "faiss",
    "sample" : None,
    "mmap" : False,
    "quantize": None

}

embeddings = np.random.rand(100,32).astype("float32")

faiss_index = Faiss(config=config,backend = None)

faiss_index.index(embeddings)

faiss_index.save("test.index")
print("Index saved as test.index") 

faiss_load = Faiss(config, None)

faiss_load.load("test.index")

print("index loaded successfully")


queries = np.random.rand(5,32).astype("float32")

results = faiss_load.search(queries,limit=3)

print ("search results:", results)