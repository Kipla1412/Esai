
import os
import json
import tempfile

from ..ann import ANNFactory
from ..vectors import VectorsFactory
from .index import Action, Configuration, Functions, Stream, Transform,Reducer,Indexes
from .search import Search
from ..scoring import ScoringFactory


class Embeddings:

    def __init__(self, config=None, models=None, **kwargs):

        self.config = None

        self.model =None
        self.models = models
        self.scoring = None
        self.reducer = None
        self.ann = None
        self.indexes = None
        self.graph = None

        self.ids =None
        self.database = None
        # self.function = None

        config = {**config, **kwargs} if config and kwargs else kwargs if kwargs else config

        self.configure(config)
        
    def __enter__(self):

        return self
    def __exit__(self,*args):

        self.close()

    def score(self,documents):

        if self.isweighted():
            self.scoring.index(Stream(self)(documents))

    def index(self, documents,reindex = False,checkpoint =None):

        self.initindex(reindex)

        transform =Transform(self, Action.REINDEX if reindex else Action.INDEX,checkpoint)
        stream = Stream(self, Action.REINDEX if reindex else Action.INDEX)

        with tempfile.NamedTemporaryFile(mode ="wb", suffix =".npy") as buffer:

            ids,dimensions,embeddings =transform(stream(documents),buffer)

            # return  ids,dimensions,embeddings

            if embeddings is not None:

                if self.config.get('pca'):
                    self.reducer = Reducer(embeddings,self.config('pca'))
                    self.reducer(embeddings)

                self.config["dimensions"] = dimensions

                self.ann = self.createann()
                self.ann.index(embeddings)

            #return  ids,dimensions,embeddings

        if self.issparse():
            self.scoring.index()

        if self.indexes:
            self.indexes.index()

    def upsert(self,documents,checkpoint =None):

        if not self.count():

            self.index(documents,checkpoint=checkpoint)
            return
        
        transform = Transform(self,Action.UPSERT, checkpoint = checkpoint)
        stream = Stream(self, Action.UPSERT)

        with tempfile.NamedTemporaryFile(mode="wb", suffix=".npy") as buffer:
            
            ids, _ ,embeddings =transform(stream(documents), buffer)

            if embeddings is not None:

                if self.reducer:

                    self.reducer(embeddings)

                self.ann.append(embeddings)

            if ids and not self.database:

                self.ids = self.createids(self.ids + ids)

        if self.issparse():

            self.scoring.upsert()

        if self.indexes:

            self.indexes.upsert()

    def delete(self,ids):

        indices = []

        deletes = []

        if self.ann or self.scoring:

            for uid in ids:

                indices.extend([index for index,value in enumerate(self.ids) if uid == value])

            for index in indices:

                deletes.append(self.ids[index]) 
                self.ids[index] = None

        if indices:

            if self.isdense():

                self.ann.delete(indices)

            if self.issparse():

                self.scoring.delete(indices)

            if self.indexes:

                self.indexes.delete(indices)

        return deletes

    def transform(self, document,category =None, index =None):

        return self.batchtransform([document], category,index)[0] 

    def batchtransform(self,documents,category =None, index =None):

        self.defaults()
        model = self.findmodel(index) 

        embeddings = model.batchtransform(Stream(self)(documents), category)

        if self.reducer:

            self.reducer(embeddings)

        return embeddings
    
    def search(self,query,limit =None,weights =None,index =None,parameters =None,graph =False):

        results =self.batchsearch([query],limit,weights,index,[parameters],graph)
        return results[0] if results else results
    
    
    def batchsearch(self,queries,limit =None,weights =None,index =None,parameters =None, graph =False):
        
        graph = graph if self.graph else False
        results = Search(self, indexids = graph)(queries,limit,weights,index,parameters)

        return [self.graph.filter(x) if isinstance(x,list) else x for x in results] if graph else results


    def configure(self,config):

        self.config = config

        scoring = self.config.get("scoring") if self.config else None
        self.scoring = self.createscoring() if scoring and not self.hassparse() else None
        self.model = self.loadvectors() if self.config else None

    def initindex(self,reindex):

        self.defaults()

        self.ids = None

        if self.ann:

            self.ann.close()
        
        self.ann = None

        if self.hassparse():

            self.scoring = self.createscoring()

        self.indexes = self.createindexes()

    def defaults(self):

        self.config = self.config if self.config else {}
        
        if not self.config.get("scoring") and any(self.config.get(key) for key in ["keyword","sparse","hybrid"]):
            self.defaultsparse()

        if not self.model and (self.defaultallowed() or self.config.get("dense")):
            self.config["path"] = "sentence-transformers/all-MiniLM-L6-v2"

        self.model = self.loadvectors()

    def findmodel(self,index = None):

        return(
            self.indexes.findmodel(index)
            if index and self.indexes
            else(
                self.model
                if self.model
                else self.scoring.findmodel() if self.scoring and self.scoring.findmodel() else self.indexes.findmodel if self.indexes else None
                
            )
        )


    def defaultallowed(self):

        params =[("keyword" ,False),("sparse",False),("defaults",True) ]
        return all(self.config.get(key,default) == default for key ,default in params)
    
    def loadvectors(self):

        if "indexes" in self.config and  self.models is None:
            self.models = {}
        
        dense = self.config.get("dense")

        if not self.config.get("path") and dense and isinstance(dense,str):

            self.config["path"] = dense

        return VectorsFactory.create(self.config,self.scoring,self.models)
    
    def defaultsparse(self):

        method = None
        for x in ["keyword","hybrid"]:
            value = self.config.get(x)

            if value:
                method =value if isinstance(value,str) else "bm25"

                if x == "hybrid":
                    self.config["dense"] = True

        sparse = self.config.get("sparse", {})
        if sparse or method == "sparse":

            sparse ={"path": self.config.get("sparse")} if isinstance(sparse, str) else {} if isinstance(sparse, bool) else sparse
            sparse["path"] = sparse.get("path","opensearch-project/opensearch-neural-sparse-encoding-doc-v2-mini")

            self.config["scoring"] = {**{"method":"sparse"}, **sparse}

        elif method:

            self.config["scoring"] = {"method":method, "terms":True, "normalize":True}
        
    def createann(self):

        if self.ann :
            self.ann.close()

        return ANNFactory.create(self.config) if self.config.get('path') or self.defaultallowed() else None
     
    def isdense(self):
        return self.ann is not None
    
    def issparse(self):
        return self.scoring and self.scoring.issparse()
    
    def isweighted(self):

        return self.scoring and self.scoring.isweighted()
    
    def count(self):

        if self.ann:
           return self.ann.count()
        
        if self.scoring:

            return self.scoring.count()
        if self.ids:

            return len(uid for uid in self.ids if uid is not None)
        
        return 0 
    
    def createscoring(self):

        if self.scoring:
            self.scoring.close()

        if "scoring" in self.config:
            config = self.config["scoring"]
            config = config if isinstance(config,dict) else {"method" : config}

            config = self.columns(config)
            return ScoringFactory.create(config, self.models)
        
        return None
    
    def hassparse(self):
        return ScoringFactory.issparse(self.config.get("scoring"))
    
    def createindexes(self):

        if self.indexes:
            self.indexes.close()

        if "indexes" in self.config:

            indexes = {}
            for index,config in self.config["indexes"].items():

                indexes[index] =Embeddings(config,model = self.model)

            return Indexes(self,indexes)
        
        return None


    def columns(self,config):

        if "columns" in self.config :

            config = config.copy()
            config["columns"] = self.config["columns"]

        return config  
        
    def close(self):

        self.ids =None
        self.config =None

        if self.ann:
            self.ann.close()
            self.ann = None
        
        if self.scoring:
            self.scoring.close()
            self.scoring = None

        if self.indexes:

            self.indexes.close()
            self.indexes =None

        if self.model:
            self.model.close()
            self.model = None
        self.models =None
