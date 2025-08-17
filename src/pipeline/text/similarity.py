import numpy as np
from .crossencoder import CrossEncoder
from .labels import Labels

class Similarity(Labels):

    def __init__(self,path =None,quantize =False,gpu =True,model = None,dynamic =True ,crossencoder =False ,**kwargs):

        super().__init__(path,quantize,gpu,model,False if crossencoder else dynamic, **kwargs)
        self.crossencoder =CrossEncoder(model = self.pipeline) if crossencoder else None

    def __call__(self ,query,texts,multilabel = True,**kwargs):

        if self.crossencoder:

            return self.crossencoder(query,texts,multilabel)
        
        scores =super().__call__(texts,[query] if isinstance(query,str) else query,multilabel,**kwargs)

        scores = [[score for _,score in sorted(row)] for row in scores]

        scores = np.array(scores).T.tolist()

        scores = [sorted(enumerate(row), key=lambda x:x[1],reverse=True) for row in scores]

        return scores[0] if isinstance(query,str) else scores




