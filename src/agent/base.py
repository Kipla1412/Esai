
from .factory import ProcessFactory

class Agent:

    def __init__(self,**kwargs):

        if "max_iterations" in kwargs:
            kwargs["max_steps"] = kwargs.pop("max_iterations")

        self.process =ProcessFactory.create(kwargs)
        self.tools = self.process.tools

    def __call__(self,agent_name,text,maxlength =8192,stream =False,**kwargs):

        self.process.model.parameters(maxlength)

        return self.process.run(text,stream =stream,**kwargs)

    

