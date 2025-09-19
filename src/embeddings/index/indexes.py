import os
from .documents import Documents

class Indexes:

    def __init__(self,embeddings,indexes):

        self.embeddings = embeddings
        self.indexes =indexes

        self.documents =None
        self.checkpoint = None

        columns= embeddings.config.get("columns",{})
        self.text = columns.get("text","text")
        self.object = columns.get("object","object")

        self.indexing = embeddings.model or embeddings.scoring

    def __contains__(self,name):

        return name in self.indexes
    
    def __getitem__(self,name):
        return self.indexes[name]
    
    def __getattr__(self,name):

        try:
            return self.indexes[name]
        except Exception as e:
            raise AttributeError from e
        
    def default(self):

        return list(self.indexes.keys())[0]
    
    def findmodel(self,index=None):

        matches =[self.indexes[index].findmodel()]if index else [index.findmodel() for index in self.indexes.values() if index.findmodel()]
        return matches[0] if matches else None
        
    def insert(self,documents ,index =None, checkpoint =None):

        if not self.documents:
           self.documents = Documents()
           self.checkpoint = None

        batch =[]
        for _, document, _ in documents:
            
            parent = document

            if isinstance(parent, dict):
                parent = parent.get(self.text, document.get(self.object))

            if parent is not None and not self.indexing:
                     
                batch.append((index,document,None))
                index += 1

        self.documents.add(batch)

    def delete(self, ids):

        for index in self.indexes.values():
            index.delete(ids)

    def index(self):

        for name,index in self.indexes.items():

            index.index(self.documents, f"{self.checkpoint}/{name}" if self.checkpoint else None)

            self.documents.close()
            self.documents = None
            self.checkpoint = None

    def upsert(self):

        for index in self.indexes.values():
            index.upsert(self.documents)

            self.documents.close()
            self.documents = None

    def save(self,path):

        for name,index in self.indexes.items():

            index.save(os.path.join(path,name))

    def load(self,path):

        for name,index in self.indexes.items():

            directory = os.path.join(path,name)

            if index.exists(directory):

                index.load(directory)

    def close(self):

        for index in self.indexes.values():

            index.close()
                