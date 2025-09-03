from ..util import Resolver
from .bm25 import BM25
from .tfidf import TFIDF

class ScoringFactory:
    
    @staticmethod
    def create(config,model =None):

        scoring =None

        if isinstance(config, str):

            config = {"method" : config}