import datetime
import platform

from ..version import __version__

class ANN:

    def __init__(self,config):

        self.backend = None
        self.config = config

    def load(self,path):

        raise NotImplementedError
    
    def index(self,embeddings):

        raise NotImplementedError
    
    def append(self,embeddings):

        raise NotImplementedError
    
    def delete(self,ids):

        raise NotImplementedError
    
    def search(self,queries,limit):

        raise NotImplementedError
    
    def count(self):

        raise NotImplementedError
    
    def save (self,path):

        raise NotImplementedError
    
    def setting(self,name,default = None):

        backend = self.config.get(self.config["backend"])

        setting = backend.get(name) if backend else None
        return setting if setting else default
    
    def metadata(self,settings = None):

        create = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
       
        if settings:
            self.config["build"] = {

                "create" : create,
                "python": platform.python_version(),
                "settings": settings,
                "system": f"{platform.system()} ({platform.machine()})",
                "txtai": __version__,
            }
    def close(self):
        self.backend = None
