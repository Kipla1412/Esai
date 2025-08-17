
from .generation import Generation
from .huggingface import HFGeneration
from .litellm import LiteLLM
from  ...util import Resolver 

class  GenerationFactory:
    
    @staticmethod
    def create(path,method,kwargs):

        method = GenerationFactory.method(path,method)

        if method == "litellm":
            return LiteLLM(path,**kwargs)
        
        if method == "transformers":
            return HFGeneration(path,**kwargs)
        else:

            return GenerationFactory.resolve(path,method,**kwargs)
        
    @staticmethod
    def method(path,method):

        if not method :

            if LiteLLM.ismodel(path):
                method = "litellm"

            else:
                method ="transformers"

        return method
    
    @staticmethod
    def resolve(path,method,**kwargs):

        try:
            return Resolver()(method)(path, **kwargs)
        except Exception as e:
            raise ImportError(f"Unable to resolve generation framework: '{method}'") from e

