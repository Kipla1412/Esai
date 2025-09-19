from ..app import Application

class API(Application):

    def __init__(self,config,loaddata=False):
        super().__init__(config, loaddata)