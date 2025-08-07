
from zipfile import BadZipFile
try:

   import  skops.io as sio 
   from sklearn.decomposition import TruncatedSVD

   REDUCER = True
except ImportError:
   REDUCER =False 

from ...serialize import SerializeFactory

class Reducer:
   
   def __init__(self,embeddings =None,components =None):
      
      if not REDUCER:
            raise ImportError('Dimensionality reduction is not available - install "vectors" extra to enable')
      
      self.model = self.build(embeddings,components) if embeddings is not None and components else None

   
   def __call__(self, embeddings):
        
        pc = self.model.components_
        factor = embeddings.dot(pc.transpose())

        
        if pc.shape[0] == 1:
            embeddings -= factor * pc
        elif len(embeddings.shape) > 1:
            
            for x in range(embeddings.shape[0]):
                embeddings[x] -= factor[x].dot(pc)
        else:
           
            embeddings -= factor.dot(pc)
       
   def build(self,embeddings,components):
       
       model =TruncatedSVD(n_components = components,random_state = 0)

       model.fit(embeddings)

       return model
   
   def load(self,path):
       
       try:       
           self.model  = sio.load(path)
       except(BadZipFile,KeyError):
           self.model =SerializeFactory.create("pickle").load(path)

   def save(self,path):
       
       sio.dump(self.model,path)
       
           
      
      
