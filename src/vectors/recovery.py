
import os
import shutil

class Recovery :

    def __init__(self,checkpoint,vectorsid,load):

        self.spool,self.path,self.load = None,None,load

        path =f"{checkpoint}"/{vectorsid}

      