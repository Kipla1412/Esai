from transformers.utils import cached_file

try :

    import litellm as api

    LITELLM =True
except ImportError:

   LITELLM = False
from .generation import Generation

class LiteLLM(Generation):

    @staticmethod
    def ismodel(path):

        if isinstance(path,str) and LITELLM:

            debug = api.suppress_debug_info

            try:

                api.suppress_debug_info = True
                return api.get_provider_llm(path) and not LiteLLM.ishub(path)
            except:
                return False
            finally:

                api.suppress_debug_info = debug
        return False
    
    @staticmethod

    def ishub(path):

        try:
            return cached_file(path_or_repo_id=path, file_name = "config.json") is not None if "/" in path else False
        except:

            return False
        
    def __init__(self, path,template = None,**kwargs):
        super().__init__(path,template,**kwargs)

        if not LITELLM:
            raise ImportError('LiteLLM is not available - install "pipeline" extra to enable')
        
        self.kwargs = {k:v for k,v in self.kwargs.items() if k not in ["quantize", "gpu", "model", "task"]}

    def stream(self,texts,maxlength,stream,stop,**kwargs):

        for text in texts:

            result = api.completion(
                model = self.path,
                messages = [{"content": text, "role": "prompt"}] if isinstance(text,str) else text,
                max_tokens =maxlength,
                stream = stream,
                stop =stop,
                **{**self.kwargs,**kwargs}

            )

            yield from self.response(result if stream else [result])
    