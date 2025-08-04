

from .transform import Action
from .autoid import AutoID

class Stream:

    def __init__(self,embeddings,action =None):

        self.embeddings = embeddings
        self.action = action

        self.config = embeddings.config
        self.offset = self.config.get("offset", 0) if self.action == Action.UPSERT else 0

        autoid = self.config.get("autoid",self.offset) 

        autoid = 0 if isinstance(autoid,int) and action != Action.UPSERT else autoid
        self.autoid = AutoID(autoid)

    def __call__(self,documents,checkpoint =None):

        for document in documents:

            if isinstance(document,dict):

                document = document.get("id"), document, document.get("tags")
            
            elif isinstance(document,tuple):

                document = document if len(document) >= 3 else (document[0], document[1], None)
            else:

                document = None,document,None

            if self.action and  document[0] is None:
                
                document = (self.autoid(document[1]), document[1], document[2])

            yield document

        current = self.autoid.current()
        
        if self.action and current:

            self.config["autoid"] = current


         



            