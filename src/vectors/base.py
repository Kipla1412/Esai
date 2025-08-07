import json
import os
import tempfile
import uuid
import numpy as np
from ..pipeline import Tokenizer

#from .dense import sbert
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
            #self.dimensionality = self.config.get("dimensionality")

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
    
    def vectors(self,documents,batchsize = 500,checkpoint =None,buffer = None,dtype =None):

        ids,dimensions,batches,stream = self.index(documents,batchsize,checkpoint)

        embeddings =None
        if ids:
            embeddings =np.memmap(buffer,dtype=dtype,shape=(len(ids),dimensions),mode ="w+")

            x = 0
            with open(stream,"+rb") as queue :
                for _ in range(batches):
                    batch = self.loadembeddings(queue)
                    embeddings[x : x + batch.shape[0]] = batch
                    x += batch.shape[0]

        if not checkpoint:
            
            os.remove(stream)

        return (ids,dimensions,embeddings)

    def index(self,documents,batchsize =500,checkpoint =None):

        ids,dimensions,batches,stream = [],None,0,None

        vectorsid = self.vectorsid() if checkpoint else None
        recovery = Recovery(checkpoint,vectorsid,self.loadembeddings) if checkpoint else None

        with self.spool(checkpoint,vectorsid) as output:

            stream = output.name

            batch =[]

            for document in documents:
                batch.append(document)

                if len(batch) == batchsize :

                    uids,dimensions = self.batch(batch,output,recovery)

                    ids.extend(uids)

                    batches += 1

                    batch =[]

                if batch:

                    uids,dimensions =self.batch(batch,output,recovery)
                    ids.extend(uids)

                    batches += 1

            return (ids,dimensions,batches,stream)
        

    def vectorsid(self):

        select =["path","method","tokenizer", "maxlength", "tokenize", "instructions", "dimensionality", "quantize"]
        config ={k:v for k,v in self.config.items() if k in select}

        config.update(config.get("vectors",{}))
        return str(uuid.uuid5(uuid.NAMESPACE_DNS,json.dumps(config,sort_keys= True)))
    
    def spool(self,checkpoint,vectorsid ):

        if checkpoint :

            os.makedirs(checkpoint,exist_ok=True)

            return open(f"{checkpoint}/vectorsid","wb")
        
        return tempfile.NamedTemporaryFile(mode ="wb",suffix =".npy", delete = False )
    
    def batch(self,documents,output,recovery) :
        # extract ids and clean input

        ids = [uid for uid,_,_ in documents]
        documents =[self.prepare(data,"data") for _,data,_ in documents]

        embeddings = recovery() if recovery else None

        embeddings = self.vectorize(documents,"data") if embeddings is None else embeddings

        if embeddings is not None :

            dimensions = embeddings.shape[1]
            self.saveembeddings(output,embeddings)

        return (ids,dimensions)
    
    def saveembeddings(self,f,embeddings):

        return np.save(f,embeddings,allow_pickle=False)
           
    
    def loadembeddings(self,f):

        return np.load(f,allow_pickle= False)
    
    def dot(self,queries,data):
        
        return np.dot(queries,data.T).tolist()
    
    def close(self):

        self.model =None


        
   






    

        
