
from .messagepack import MessagePack
from .pickle import Pickle

class SerializeFactory:

    @staticmethod
    def create(method =None,**kwargs):

        if method == "pickle":

            return Pickle(**kwargs)
        
        return MessagePack(**kwargs)