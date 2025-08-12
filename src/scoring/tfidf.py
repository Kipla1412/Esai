
from .base import Scoring
from .terms import Terms
from ..pipeline import Tokenizer
from collections import Counter
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

        documents = {} if self.config.get("content") else None

        #6) Normalization

        self.normalize = self.config.get("normalize")
        self.avgscore = None

    def insert(self,documents,index = None, checkpoint = None):

        for uid,document,tags in documents:

            if isinstance(document,dict):

                document = self.document.get(self.text, self.document.get(object))

            if document is not None:

                uid = index if index is not None else uid

                if isinstance(document(str,list)):

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
        
    def loadtokenizer(self,text):

        if self.config.get("tokenizer"):

            return Tokenizer(**self.config.get("tokenizer"))

        if self.config.get("terms"):

            return Tokenizer()
        
        return Tokenizer.tokenize(text)
    
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

            for x ,word in self.docfreq.items():
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
