import os
import yaml

from multiprocessing.pool import ThreadPool

from ..pipeline import PipelineFactory
from ..agent import Agent

class Application:
    @staticmethod
    def read(data):
        if isinstance(data, str):
            if os.path.exists(data):
                with open(data, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f)                    #parse the yaml content into dict
            
            data = yaml.safe_load(data)
            if not isinstance(data, str):
                return data
            
            raise FileNotFoundError(f"unable to load the file: '{data}'")
        
        return data
    
    def __init__(self, config, loadata=False):
        self.config = Application.read(config)
        self.pool = None

        self.createpipelines()
        
        self.createagents()


    def createpipelines(self):
        self.pipelines = {}

        pipelines = list(PipelineFactory.list().keys())             #default pipelines

        for key in self.config:
            if "." in key:                                          # custom pipelines
                pipelines.append(key)

        for pipeline in pipelines:
            if pipeline in self.config:
                config = self.config[pipeline] if self.config[pipeline] else {}

                self.pipelines[pipeline] = PipelineFactory.create(config, pipeline)


    def createagents(self):
        self.agents = {}

        if "agent" in self.config:
            for agent, config in self.config["agent"].items():
                config = config.copy()

                config["llm"] = self.function("llm")

                for tool in config.get("tools", []):
                    if isinstance(tool, dict) and "target" in tool:
                        tool["target"] = self.function(tool["target"])
                    
                self.agents[agent] = Agent(**config)

    def resolvetask(self, task):
        """
        we pass the task as an input and 
        return the task as a callable object dictionary
        args: {"action": "tabular"}
        returns: {"action": <tabualr object>}

        task = {"action": "tabular"}
        action = "tabular"
        values = ["tabular"]
        actions = [Tabular()]
        """
        task = {"action": task} if isinstance(task, (str, list)) else task   #we need dict form: {"action": "tabular"}

        if "action" in task:
            action = task["action"]
            values = [action] if not isinstance(action, list) else action

            actions = []
            for a in values:
                actions.append(self.function(a))
            
            task["action"] = actions[0] if not isinstance(action, list) else actions        #return the values in their original form

        return task
    
    def function(self, function):
        """
        return the function as callable object.

        function: values from resolvetask

        if the value is not in pipeline and workflow, we have to create it.
        """
        if function in self.pipelines:
            return self.pipelines[function]     # callable object
        
        
        return PipelineFactory.create({}, function)

    def pipeline(self, name, *args, **kwargs):
        args = args[0] if args and len(args) == 1 and isinstance(args[0], tuple) else args
        
        if name in self.pipelines:
            return self.pipelines[name](*args, **kwargs)
        
        return None
    
    
    def agent(self, name, *args, **kwargs):
        if name in self.agents:
            return self.agents[name](*args, **kwargs)
        
        return None

    def wait(self):
        """
        Closes threadpool and waits for completion.
        """

        if self.pool:
            self.pool.close()
            self.pool.join()
            self.pool = None