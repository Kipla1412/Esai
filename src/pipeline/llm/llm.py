import logging
from ..base import Pipeline
from .factory import GenerationFactory

logger =logging.getLogger(__name__)

class LLM(Pipeline):

    def __init__(self,path=None,method =None,**kwargs):

        path = path if path else "google/flan-t5-base"

        self.generator = GenerationFactory.create(path, method,**kwargs)

    def __call__(self,text,maxlength = 500, stream = False,stop =None, defaultrole ="prompt", stripthink =False, **kwargs):

        logger.debug(text)

        return self.generator(text,maxlength,stream,stop,defaultrole,stripthink,**kwargs)
    
    def isvision(self):

        return self.generator.isvision()
