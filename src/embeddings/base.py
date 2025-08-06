
import os
import json
import tempfile

from ..vectors import VectorsFactory
from .index import Action, Configuration, Functions, Stream, Transform


class Embeddings:

    def __init__(self, config=None, models=None, **kwargs):

        self.config = None

        self.model =None
        self.models = models
        self.scoring = None

        # self.ids =None
        # self.function = None

        config = {**config, **kwargs} if config and kwargs else kwargs if kwargs else config

        self.configure(config)
        
    def __enter__(self):

        return self
    def __exit__(self,*args):

        self.close()


    def index(self, documents,reindex = False,checkpoint =None):

        self.initindex(reindex)

        transform =Transform(self, Action.REINDEX if reindex else Action.INDEX,checkpoint)
        stream = Stream(self, Action.REINDEX if reindex else Action.INDEX)

        with tempfile.NamedTemporaryFile(mode ="wb", suffix =".npy") as buffer:

            ids,dimensions,embeddings =transform(stream(documents),buffer)

        return  ids,dimensions,embeddings

    def configure(self,config):

        self.config = config
        self.model = self.loadvectors() if self.config else None

    def initindex(self,reindex):

        self.defaults()
        

    def defaults(self):

        config = self.config if self.config else {}

        if not self.model and (self.defaultallowed() or self.config.get("dense")):
            self.config["path"] = "sentence-transformers/all-MiniLM-L6-v2"

        self.model = self.loadvectors()

    def defaultallowed(self):

        params =[("keyword" ,False),("sparse",False),("defaults",True) ]
        return all(self.config.get(key,default) == default for key ,default in params)
    
    def loadvectors(self):
        
        dense = self.config.get("dense")

        if not self.config.get("path") and dense and isinstance(dense,str):

            self.config["path"] = dense

        return VectorsFactory.create(self.config,self.scoring,self.models)
    
    def close(self):

        if self.model:
            self.model.close()
            self.model = None
        self.models =None
