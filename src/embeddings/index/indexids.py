from ...serialize import Serializer

class IndexIds:

    def __init__(self,embeddings,ids =None):

        self.config = embeddings.config
        self.ids = ids

    def __iter__(self):

        yield from self.ids

    def __getitem__(self,index):

        return self.ids[index]
    
    def __setitem__(self,index,value):

        self.ids[index] = value

    def __add__(self,ids):

        self.ids + ids
    
    def load(self,path):

        if "ids" in self.config:

            self.ids = self.config.pop("ids")

        else:
            self.ids = Serializer.load(path)

    def save(self,path):
        
        Serializer.save(path)