
import inspect
from transformers import pipeline

from ..models import Models
from ..util import Resolver

from .tensors import Tensors

class HFPipeline(Tensors):

    def __init__ (self, task, path = None, quantize =False, gpu =False, model =None, **kwargs):

        if model:

            self.pipeline = model.pipeline if isinstance (model,HFPipeline) else model

        else:
            deviceid = Models.deviceid(gpu) if "device_map" not in kwargs else None
            device = Models.device(deviceid) if deviceid is not None else None

            modelargs, kwargs = self.parseargs(**kwargs)

            if isinstance (path, (list, tuple)):

                config = path[1] if path[1] and  isinstance(path[1], str) else None


                model = Models.load(path[0], config, task)

                self.pipeline = pipeline(task, model =model, tokenizer = path[1], device =device,model_kwargs=modelargs, **kwargs)

            else :
                self.pipeline = pipeline(task, model =path, device =device, model_kwargs = modelargs, **kwargs) 

            if deviceid == -1 and quantize:

                self.pipeline.model = self.quantize(self.pipeline.model) 
            
        Models.checklength(self.pipeline.model, self.pipeline.tokenizer)

    def parseargs(self, **kwargs):

        args = inspect.getfullargspec(pipeline).args

        dtype = kwargs.get("torch_dtype")

        if dtype and isinstance (dtype, str) and  dtype !=  "auto":
            kwargs["torch_dtype"] = Resolver()(dtype)
        return ({arg: value for arg, value in kwargs.items() if arg not in args}, {arg: value for arg, value in kwargs.items() if arg in args})
    
    def maxlength(self):

        return Models.maxlength(self.pipeline.model , self.pipeline.tokenizer)

