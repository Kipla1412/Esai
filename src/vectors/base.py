import json
import os
import tempfile
import numpy as np
from ..pipeline import Tokenizer

from .dense import factory,sbert
from .recovery import Recovery

class Vectors : 

    def __init__(self,config,scoring,models):

        self.config = config
        self.scoring =scoring
        self.models = models
        self.tokenize =False

        if config :

            self.model = self.load(config.get("path"))
            self.encodebatch = self.config.get("encodebatch", 32)
            self.dimensionality = self.config.get("dimensionality")

    def loadmodel(self,path):

        return NotImplementedError
    
    def encode(self,data,category=None):

        return NotImplementedError
    
    def load(self,path):

        if self.models and path in self.models:
            return self.models[path]
        
        model =self.loadmodel(path)

        if self.models is not None and path:

            self.models[path] =model

        return model
    
    def transform(self,documents):

        return self.batchtransform([documents])[0]
    
    def batchtransform(self,documents,category =None):

        documents =[self.prepare(data,category) for _,data,_ in documents]

        if documents and isinstance(documents[0], np.ndarray):

            return np.array(documents[0],dtype=np.float32)
        
        return self.vectorize(documents,category)
    
    def prepare(self,data,category =None):

        data = self.tokens(data)

        category= category if category else "query"

        return data
    
    def tokens(self, data):

        if self.tokenize and isinstance(data,str):
            data =Tokenizer.tokenize(data)

        return data
    
    def vectorize(self,data,category =None):

        category = category if category else "query"

        embeddings = self.encode(data,category)

        if embeddings is not None:

            embeddings = self.normalize(embeddings)

        return embeddings
        
    def normalize(self,embeddings):
        
        if len(embeddings.shape)>1:

            embeddings /= np.linalg.norm(embeddings,axis =1)[:,np.newaxis]

        else :

            embeddings /= np.linalg.norm(embeddings)

        return embeddings
    
    def index(self,documents,batch_size =500,checkpoint =None):

        ids,dimensions,batches,stream = [],None,0,None

        vectorsid = self.vectorsid if checkpoint else None
        recovery = Recovery(checkpoint,vectorsid,self.loadembeddings) if checkpoint else None

        with self.spool(checkpoint,vectorsid) as output:

            stream = output.name

            batch =[]

            for document in documents:
                batch.append(document)

                if len(batch) == self.batch_size :

                    uids,dimensions = self.batch(batch,output,recovery)

                    ids.extend(uids)

                    batches += 1

                    batch =[]

                if batch:

                    uids,dimensions =self.batch(batch,output,recovery)
                    ids.extend(uids)

                    batches += 1

            return (ids,dimensions,batches,stream)





    

        
