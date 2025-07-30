
import os
import shutil

class Recovery :

    def __init__(self,checkpoint,vectorsid,load):

        self.spool,self.path,self.load = None,None,load

        path =f"{checkpoint}/{vectorsid}"

        if os.path.exists(path):

            self.path = f"{checkpoint} /recovery"

            shutil.copyfile(path,self.path)

            self.spool =  open (self.path,"rb")

    def __call__(self):

        try :

            return self.load(self.spool)if self.spool else None
        except EOFError:

            self.spool.close()

            os.remove(self.path)

            self.spool,self.path = None ,None

        return None 





      