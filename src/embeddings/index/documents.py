
import os
import tempfile

from ...serialize import SerializeFactory

class Documents:

    def __init__(self):

        self.documents = None
        self.batch = 0
        self.size = 0

        self.serialize =SerializeFactory.create("pickle", allowpickle=True)

    def __len__(self):

        return self.size
    
    def __iter__(self):

        self.documents.close()

        with open(self.documents.name, "rb") as queue:
            
            for _ in range(self.batch):

                documents = self.serializer.loadstream(queue)

                yield from documents

    def add_document(self,documents):

        if not self.documents :
            self.documents = tempfile.NamedTemporaryFile(mode="wb", suffix=".docs", delete=False)

        self.serializer.savestream(documents, self.documents)
        self.batch += 1
        self.size += len(documents)

        return documents
    
    def close(self):

        os.remove(self.documents.name)

        self.documents =None
        self.batch = 0
        self.size = 0
   