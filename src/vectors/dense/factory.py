from ...util import Resolver
from .sbert import STVectors

class VectorFactory:
    
    @staticmethod
    def create(config,scoring =None,models =None):

       method = VectorFactory.create(config)

       if method == "sentence-transformers":
           
           return STVectors (config,scoring,models) if config and config("path") else None
       
       return VectorFactory.resolve(method,config,scoring,models) if method else None
    
    @staticmethod
    def method(config):

        method =config.get("method")
        path = config.get("path")

        if not method:
            if path:

                if STVectors.ismodel(path):
                     method = "transformers"

        return method
    
    @staticmethod

    def resolve(backend,config,scoring,models):

        try:
            Resolver()(backend)(config,scoring,models)

        except Exception as e:
            
            raise ImportError(f"Unable to resolve vectors backend: '{backend}'") from e



           