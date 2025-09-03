from .tfidf import TFIDF
import numpy as np

class BM25(TFIDF):

    def __init__(self,config = None):
        super().__init__(config)

        self.k1 = self.config.get("k1",1.2)
        self.b = self.config.get("b",0.75)

    def computeidf(self, freq):
        
        return np.log(1 + (self.total - freq + 0.5) / (freq + 0.5))
    
    def score(self,freq, idf, length):

        k = self.k1 * ((1 - self.b) + self.b * length / self.avgdl)
        return idf * (freq * (self.k1 + 1)) / (freq + k)
