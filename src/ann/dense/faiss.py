import math
import numpy as np
import platform

from faiss import omp_set_num_threads
from faiss import index_factory,IO_FLAG_MMAP,METRIC_INNER_PRODUCT ,read_index,write_index

from faiss import index_binary_factory,read_index_binary,write_index_binary,IndexBinaryIDMap
from ..base import ANN

if platform.system() == 'Darwin':
    omp_set_num_threads(1)
    

class Faiss(ANN):

    def __init__(self,config,backend):

        super().__init__(config)

        quantize = self.config.get("quantize")
        self.qbits = quantize if quantize and isinstance(quantize,int) and not isinstance(quantize,bool) else None

    def load(self,path):
        
        readindex = read_index_binary if self.qbits else read_index

        # load index 
        self.backend = readindex(path,IO_FLAG_MMAP if self.setting("mmap") is True  else 0)

    
    def index(self,embeddings):
        
        train, sample = embeddings , self.setting("sample")

        if sample:

            rng = np.random.default_rng(0)
            indices =sorted(rng.choice(train.shape[0] ,int(sample* train.shape[0]),replace = False,shuffle = False))
            train = train [indices]

       # configure the index 
        params = self.configure(embeddings.shape[0],train.shape[0])

        self.backend = self.create(embeddings,params)
        self.backend.train(train)

        self.backend.add_with_ids(embeddings, np.arange(embeddings.shape[0],dtype = np.int64))

        self.config["offset"] = embeddings.shape[0]

        self.metadata({"components" : params})

    def create(self,embeddings,params):

        if self.qbits:

            index = index_binary_factory(embeddings.shape[1]*8, params)

            if any(x in params for x in ["BFlat", "BHNSW"]):
                index = IndexBinaryIDMap(index)
                
            return index
        
        return index_factory(embeddings.shape[1],params,METRIC_INNER_PRODUCT)

    def append(self,embeddings):

        new = embeddings.shape[0]

        self.backend.add_with_ids(
            embeddings,
            np.arange(self.config["offset"],self.config["offset"]+new , dtype =np.int64)
        )

        self.config["offset"] += new

        self.metadata()

    def delete(self,ids):

        self.backend.remove_ids(np.array(ids, dtype =np.int64))
    
    def search(self,queries,limit):

        scores ,ids = self.backend.search(queries,limit)

        results =[]

        for x,score in enumerate(scores):
            results.append(list(zip(ids[x].tolist(), score.tolist())))

        return results
    
    def configure(self, count, train):
        
        return "IDMap,Flat"

    def save(self,path):

        writeindex = write_index_binary if self.qbits else write_index

        writeindex(self.backend,path)
 
