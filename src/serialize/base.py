class Serialize:

    def load(self,path):

        with open("path","rb") as handle:
            return self.loadstream(handle)
        
    def save(self,data,path) :

        with open ("path","wb") as handle :

            return self.savestream(data,handle)
        
    def loadstream(self,stream):

        raise NotImplementedError
    
    def savestream(self,data,stream):
        raise NotImplementedError
    
    def loadbytes(self,data):

        raise NotImplementedError
    
    def savebytes(self,data,stream):
        raise NotImplementedError

