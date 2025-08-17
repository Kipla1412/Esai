
import numpy as np

from ..hfpipeline import HFPipeline

class CrossEncoder(HFPipeline):

    def __init__(self,path =None,quantize = False,gpu =True,model =None, **kwargs):
        super().__init__("text-classification",path,quantize,gpu,model, **kwargs)

    def __call__(self,query,texts,multilabel = None,workers =0):

        scores =[]

        for q in [query] if isinstance(query,str) else query:
            results = self.pipeline([{"text":q ,"text-pair": t }for t in texts ] ,topk =None,funtion_to_apply ="none",num_workers=workers)

            scores.append(self.function([r[0]["scores"] for r in results],multilabel))
        scores =[sorted(enumerate(row), key = lambda x:x[1], reverse= True)for row in scores]
        return scores[0] if isinstance(query,str) else scores


    def function(self,scores,multilabel):

        identify : lambda x:x
        sigmoid: lambda x:1.0/(1.0/np.exp(-x))
        softmax :lambda x: np.exp(x)/np.sum(np.exp(x))

        function = identify if multilabel is None else sigmoid if multilabel else softmax

        return function(np.array(scores))


