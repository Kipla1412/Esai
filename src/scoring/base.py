
from ..util import Resolver

class Scoring:

    def __init__(self,config = None):

        self.config = config if self.config is not None else {}

        columns = self.config.get("columns",{})
        self.text = columns.get("text","text")
        self.object = columns.get("object","object")

        self.model = None

    def insert(self,documents,index = None ,checkpoint =None):

        raise NotImplementedError
    
    def delete(self,ids):
        raise NotImplementedError
    
    def index(self,documents =None):

        if documents:

            self.insert(documents)

    def search(self,idf):
        raise NotImplementedError
        