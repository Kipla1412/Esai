from src.pipeline.llm.rag import RAG
from src.scoring.bm25 import BM25

docs = [
    ("doc1", "Machine learning is amazing and widely used in AI applications.", None),
    ("doc2", "Deep learning is a subset of machine learning that deals with neural networks.", None),
    ("doc3", "Python is great for data science and has many useful libraries like pandas and numpy.", None),
    ("doc4", "Natural Language Processing (NLP) helps computers understand human language.", None),
    ("doc5", "Reinforcement learning is used in robotics and game AI to optimize decision making.", None),
    ("doc6", "Computer vision allows machines to interpret and understand visual data.", None),
    ("doc7", "Support Vector Machines are a type of supervised learning algorithm.", None),
    ("doc8", "Clustering is an unsupervised learning technique for grouping similar data points.", None),
    ("doc9", "Decision trees are easy to interpret and useful for classification problems.", None),
    ("doc10", "Generative models can create new data similar to the input dataset.",None),
]

config ={"terms":{"cachelimit": 1000000,"cutoff":1.0},"content":True,"normalize":True,"k1": 1.5,"b": 0.75}
bm25 =BM25(config=config)
bm25.insert(docs)

bm25.index()

rag =RAG( similarity=bm25, path ="facebook/bart-large-mnli", context=3)

questions =['What is machine learning?',"Tell me about Python data science"]

answers =rag(questions)

for q, a in zip(questions, answers):
    print(f"Q: {q}\nA: {a}\n")