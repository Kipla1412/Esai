 
from ...util import Resolver

from .faiss import Faiss

class ANNFactory:

    @staticmethod
    def create(config):

        ann = None
        backend =config.get ("backend","faiss")

        if backend == "faiss":
            ann = Faiss(config)
        else:

            return ANNFactory.resolve(backend,config)
        
        config["backend"] = backend
        return ann
        
    @staticmethod
    def resolve(backend,config):

        try:
            return Resolver()(backend)(config)
        
        except Exception as e:
            raise ImportError(f"Unable to resolve ann backend: '{backend}'") from e