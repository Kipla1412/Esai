
import numpy as np
from .action import Action

class Transform:

    def __init__(self,embeddings,action,checkpoint = None):

        self.embeddings = embeddings
        self.action = action 
        self.checkpoint = checkpoint

        self.scoring = embeddings.scoring if embeddings.issparse else None
        self.indexes = embeddings.indexes

        self.config = embeddings.config
        self.model = embeddings.model

        self.offset = embeddings.config.get("offset",0) if action == Action.UPSERT else 0
        self.batch = embeddings.config.get("batch",1024)
        
        quantize = embeddings.config.get("quantize")
        self.qbits = quantize  if  isinstance (quantize, int) and not isinstance(quantize,bool) else None

        columns = embeddings.config.get("columns",{})
        self.text = columns.get("text", "text")
        self.object =columns.get("object","object")

        self.indexing = embeddings.model or embeddings.scoring
        self.deletes = set()

    def __call__(self,documents,buffer):

        ids,dimensions,embeddings = None,None,None

        if self.model :

            ids, dimensions, embeddings = self.vectors(documents,buffer)

        else :

            return self.ids(documents)
        
        return (ids, dimensions, embeddings )
    
    def vectors(self,documents,buffer):

        dtype = np.uint8 if self.qbits else np.float32

        return self.model.vectors(self.stream(documents),self.batch,self.checkpoint,buffer,dtype)
    
    def ids(self,documents):

        ids = []
        for uid, _, _ in self.stream(documents):
            ids.append(uid)

        self.config["offset"] = self.offset

        return ids
    
    def stream(self,documents):

        batch,offset =[],0

        for document in documents:

            if isinstance(document[1], dict):

                if not self.indexing and not document[1].get(self.text):
                    document[1][self.text] = str(document[0])

                if self.text in document[1]:

                    yield (document[0], document[1][self.text], document[2])
                    offset += 1

                elif self.object in document[1]:

                    yield (document[0], document[1][self.object], document[2])
                    offset += 1

            else:
                yield document
                offset += 1

            batch.append(document)
            if len(batch) == self.batch:
                self.load(batch,offset)

                batch,offset =[],0

        if self.batch:

            self.load(batch,offset)

    def load(self,batch,offset):

        if self.action == Action.UPSERT:

            deletes =[uid for uid,_,_ in batch if uid not in self.deletes]

            if deletes:
                
                self.delete(deletes)

                self.deletes.update(deletes)

        if self.scoring:

            self.scoring.insert(batch,self.offset,self.checkpoint)

        if self.indexes:
            self.indexes.insert(batch, self.offset,self.checkpoint)

        self.offset += offset
        