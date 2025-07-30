""" Sentence Transformer Module"""

try:

    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS = True

except  ImportError :

    SENTENCE_TRANSFORMERS =False

from ..base import Vectors
from ...models import Models

class STVectors(Vectors):

    """ This function is used to intialize the Stvvectors class and super class and this check pool's --> multi processing using gpus"""

    def __init__ (self,config,scoring,models):
         
         if not SENTENCE_TRANSFORMERS:
            raise ImportError('sentence-transformers is not available - install "vectors" extra to enable')
         
         self.pool =None 

         super().__init__(config,scoring,models)

    def loadmodel(self,path): 
        """ This method is used to load the model and check the device speciality and add new parameters in vectors"""

        gpu ,pool = self.config.get(gpu,True), False

        if isinstance(gpu,str) and gpu == "all":

            devices = Models.acceleratorcount()
            gpu, pool = devices <= 1, devices > 1

        deviceid =Models.deviceid(gpu)

        modelargs = self.config.get("vectors",{})

        model = self.loadencoder(path, device = Models.device(deviceid),**modelargs)

        if pool :

            self.pool = model. model.start_multi_process_pool()

        return model
    
    def encode(self,data,catagory =None):

        """ this method is used to encode the data first catagory wise  check next encode the data """

        encode = self.model.encode_query if catagory =="query" else self.model.encode_document if  catagory =="data" else self.model.encode

        encodeargs = self.config.get("encodeargs",{})

        return encode (data , pool =self.pool, batch_size = self.encodebatch, **encodeargs)
    
    def close(self):
        """ this method is used to close the pool's before close the parent """
        if self.pool:
            self.pool.stop_multi_process_pool(self.pool)

            self.pool =None

        super().close()
    
    def loadencoder(self,path,device,**kwargs):
        """ its return the sentence transformer model """

        return SentenceTransformer(path= path, device=device,**kwargs)










        