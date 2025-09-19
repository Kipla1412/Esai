from ..util import Resolver
from .bm25 import BM25
from .tfidf import TFIDF

class ScoringFactory:
    
    @staticmethod
    def create(config,model =None):

        scoring =None

        if isinstance(config, str):

            config = {"method" : config}

        method = config.get("method", "bm25")

        if method == "bm25":
            scoring =BM25(config)

        elif method == "tfidf":

            scoring =TFIDF(config)

        else:

            scoring = ScoringFactory.resolve(method,config)

        config["method"] = method

        return scoring
    
    @staticmethod
    def issparse(config):

        indexes =["pgtext","sparse"]

        return config and isinstance(config,dict) and (config.get("method") in indexes or config.get("terms"))
    
    @staticmethod
    def resolve(backend,config):

        try:
            return Resolver()(backend)(config)
        except Exception as e:

            raise ImportError(f"Unable to resolve scoring backend: '{backend}'") from e