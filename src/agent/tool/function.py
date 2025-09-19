from smolagents import Tool

class FunctionTool(Tool):

    def __init__(self,config):

        """
        creates a functiontool

        Args:
         config : `name`, `description`, `inputs`, `output` and `target` configuration
        """

        self.name= config["name"]
        self.description = config["description"]
        self.inputs = config["inputs"]
        self.output_type =config.get["output",config.get("output_type","any")]
        self.target =config["target"]

        self.skip_forward_signature_validation =True

        super().__init__()

    def forward(self,*args,**kwargs):

        return self.target(*args,**kwargs)
