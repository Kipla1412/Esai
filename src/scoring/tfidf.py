import math
import os
from multiprocessing.pool import ThreadPool
from .base import Scoring
from .terms import Terms
from ..pipeline import Tokenizer
from collections import Counter
from ..serialize import Serializer
import numpy as np

class TFIDF(Scoring):

    def __init__(self,config):

        super().__init__(config)

        """ document stats,word stats,itf index , tags boosting, tokenization & terms & & documents, Normalization"""

        #1) document stats

        self.total = 0
        self.tokens = 0
        self.avgdl = 0

        #2) word stats

        self.docfreq = Counter()
        self.wordfreq = Counter()
        self.avgfreq = 0

        #3) itf index 

        self.idf ={}
        self.avgidf = 0

        #4) tags boosting

        self.tags = Counter()

        #5) tokenization & terms & & documents used for lazy indexing

        self.tokenizer = None

        self.terms = Terms(self.config["terms"], self.scoring,self.idf) if self.config.get("terms") else None

        self.documents = {} if self.config.get("content") else None

        #6) Normalization

        self.normalize = self.config.get("normalize")
        self.avgscore = None

    def insert(self,documents,index = None, checkpoint = None):

        for uid,document,tags in documents:

            if isinstance(document,dict):

                document = self.document.get(self.text, self.document.get(object))

            if document is not None:

                uid = index if index is not None else uid

                if isinstance(document,(str,list)):

                    if self.documents is not None:
                            
                        self.documents[uid] = document

                        tokens =self.tokenize(document) if isinstance(document,str) else document

                    if self.terms is not None:
                        self.terms.insert(uid,tokens)

                    self.addstats(tokens,tags)

                index = index + 1 if index is not None else None

    def tokenize(self,text):
        
        if not self.tokenizer:

            self.tokenizer = self.loadtokenizer()

        return self.tokenizer(text)
        
    def loadtokenizer(self):

        if self.config.get("tokenizer"):

            return Tokenizer(**self.config.get("tokenizer"))

        if self.config.get("terms"):

            return Tokenizer()
        
        return Tokenizer.tokenize
    
    def addstats(self,tokens,tags):

        self.wordfreq.update(tokens)
        self.docfreq.update(set(tokens))

        if tags:

            self.tags.update(tags.split())

        self.total += 1

    def delete(self, ids):
        
        if self.terms:
            self.terms.delete(ids)

        if self.documents:
            for uid in ids:
                self.documents.pop(uid)

    def index(self,documents = None):

        super().index(documents)

        if self.wordfreq:
            self.tokens = sum(self.wordfreq.values())

            self.avgfreq = self.tokens/len(self.wordfreq.values())
            self.avgdl = self.tokens/self.total

            #idf calculations
            idfs = self.computeidf(np.array(list(self.docfreq.values())))

            for x ,word in enumerate(self.docfreq):
                self.idf[word] = float(idfs[x])

            self.avgidf = float(np.mean(idfs))

            self.avgscore = self.score(self.avgfreq, self.avgidf,self.avgdl)

            self.tags = Counter({
                tag : number
                for tag,number in self.tags.items()
                if number >= self.total *0.005

            })

        if self.terms:

            self.terms.index()

    def computeidf(self,freq):

        return np.log((self.total + 1)/(freq +1)) + 1
    
    def score(self,freq,idf,length):

        return idf* np.sqrt(freq) * (1/np.sqrt(length))

    def search(self,query,limit = 3):

        if self.terms:

            query = self.tokenize(query) if isinstance(query,str) else query

            scores = self.terms.search(query,limit)

            if self.normalize and scores:
                maxscore =min(scores[0][1]+self.avgscore , 6*self.avgscore)

                scores = [(x ,min(score/maxscore ,1.0)) for x,score in scores]

            return self.results(scores)
        
        return None
    
    def results(self,scores):

        scores = [(x , float(score)) for x , score in scores]

        if self.documents:
            return [
                {
                    "id" : x, "text":self.documents[x], "score" : score
                }
                for x,score in scores
            ]
        return scores
    
    def batchsearch(self,queries,limit =3,threads = True):
        threads =math.ceil(self.count()/25000) if isinstance(threads, bool) and threads else int(threads)
        threads = min(max(threads,1), os.cpu_count())
        results =[]


        with ThreadPool(threads) as pool:
            for result in pool.starmap(self.search,[(x,limit) for x in queries ]):
                results.append(result)

        return results
    
    def count(self):
        return self.terms.count() if self.terms else self.total
    
    def weights(self,tokens):

        length =len(tokens)
        freq =self.computefreq(tokens)
        freq= np.array([freq[token] for token in tokens])

        idf = np.array([self.idf[token] if token in self.idf else self.avgidf for token in tokens])
        weights =self.score(freq,idf,length).tolist()

        if self.tags:
            tags ={token: self.tags[token] for token in tokens if token in self.tags }

            if tags:
                maxWeight =max(weights)
                maxTag = max(tags.values())
                weights =[max(maxWeight * (tags[tokens[x]] / maxTag), weight) if tokens[x] in tags else weight for x, weight in enumerate(weights)]
        return weights
    
    def computefreq(self,tokens):
        return Counter(tokens)
    
    def load(self,path):

        state =Serializer.load(path)
        
        for key in["docfreq","wordfreq","tags"]:
            state[key]= Counter(state[key])

        state['documents'] = dict(state['documents']) if state["documents"] else state["documents"]

        self.__dict__.update(state)

        if self.terms:

            self.terms =Terms(self.config["terms"],self.score, self.idf)
            self.terms.load(path +".terms")

    def save(self,path):

        skipfields ={"config","terms","tokenizer"}
        state ={key: value  for key, value in self.__dict__.items() if key not in skipfields}

        state["documents"] = list(state["documents"].items()) if state["documents"] else state["documents"]

        Serializer.save(state,path)
        if self.terms:
            self.terms.save(path+".terms")

    def issparse(self):
        return self.terms is not None
    
    def isnormalized(self):
        return self.normalize
    
    def close(self):
        if self.terms:
            self.terms.close()

