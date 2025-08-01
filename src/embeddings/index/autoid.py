""" Auto Id Module """
import inspect
import uuid

class AutoID:

    def __init__(self,method =None):

        self.method,self.function,self.value =None,None,None

        if not method or isinstance(method,int):
            self.method = self.sequence
            self.value =  method if method else 0

        else :
            self.method = self.uuid
            self.function = getattr(uuid,method)

        args = inspect.getfullargspec(self.function).args if self.function else []
        self.deterministic = "namespace" in args

    def __call__(self,data=None):

        return self.method(data)
    
    def sequence(self,value):

        value = self.value
        self.value+=1
        return value
    
    def uuid(self,data):

        uid = self.function(uuid.NAMESPACE_DNS,str(data)) if self.deterministic else  self.function ()

        return str(uid)
    
    def current(self):

        return self.value



        


