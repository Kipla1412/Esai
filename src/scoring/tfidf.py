
from .base import Scoring
from .terms import Terms
from collections import Counter

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

        pass

       