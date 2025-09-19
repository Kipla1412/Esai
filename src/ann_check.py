
from src.ann.dense.faiss import Faiss
import numpy as np

config ={

    "backend":"faiss",
    "quantize" : False,
    "dimensions": 16,
    "sample":None,
    "nprobe" : 4
}

faiss_index = Faiss(config)

embeddings = np.random.rand(100,config["dimensions"]).astype("float32")
faiss_index.index(embeddings)

print("Index build with vectors :" ,faiss_index.count())


queries = embeddings[:4]
results = faiss_index.search(queries, limit =3)

print(results)

print("Search results(top 3)")

for i,res in enumerate(results):
    print(f"Query{i} ---> {res}")

for i, res in enumerate(results):
    scores =[score for _, score in res]
    print(f"Query{i} normalized_scores:",faiss_index.scores(np.array(scores)))

new_embeddings = np.random.rand(10, config["dimensions"]).astype("float32")
faiss_index.append(new_embeddings)

print("after add the new embeddings counts :", faiss_index.count())

delete_ids =[1,2]
faiss_index.delete(delete_ids)
print("After delete, total vectors:",faiss_index.count())

faiss_index.save("faiss.index")
print("saved")


