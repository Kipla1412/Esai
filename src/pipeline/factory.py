import sys
import inspect 
import types

from ..util import Resolver
from .base import Pipeline

class PipelineFactory:

    @staticmethod
    def get(pipeline):

        if "." not in pipeline:
            return PipelineFactory.list()[pipeline]
        return Resolver()(pipeline)
    
    @staticmethod
    def create(config,pipeline):

        pipeline = PipelineFactory.get(pipeline)

        return pipeline if isinstance(pipeline,types.FunctionType) else pipeline(**config)
    
    @staticmethod
    def list():

        pipelines ={}
        pipeline = sys.modules[".".join(__name__.split(".")[:-1])]

        for x in inspect.getmembers(pipeline, inspect.isclass):
            if issubclass(x[1],Pipeline) and [y for y,_ in inspect.getmembers(x[1], inspect.isfunction) if y =="__call__"]:

                pipelines[x[0].lower()] =x[1]

        return pipelines


