from .errors import SerializeError
from .factory import SerializeFactory

class Serializer:

    @staticmethod
    def load(path):

        try:

            return SerializeFactory.create().load(path)
        
        except :

            return SerializeFactory.create("pickle").load(path)
        

    def save(data,path):

        SerializeFactory.create().save(data,path)
