import inspect

from types import FunctionType,MethodType

from smolagents import PythonInterpreterTool,Tool,tool as CreateTool,VisitWebpageTool,WebSearchTool
from transformers.utils import chat_template_utils ,TypeHintParsingException

from ...embeddings import Embeddings
from .embeddings import EmbeddingsTool
from .function import FunctionTool

class ToolFactory:

    DEFAULTS ={"python":PythonInterpreterTool(),"websearch":WebSearchTool(),"webview":VisitWebpageTool()}

    @staticmethod
    def create(config):

        tools = []
        for tool in config.pop("tools", []):
        
            if not isinstance(tool, Tool) and (isinstance(tool, (FunctionType, MethodType)) or hasattr(tool, "__call__")):
                tool = ToolFactory.createtool(tool)

            elif isinstance(tool, dict):

                target = tool.get("target")

                tool = (

                    EmbeddingsTool(tool)
                    if isinstance(target,Embeddings) or any(x in tool for x in ["container", "path"])
                    else ToolFactory.createtool(target,tool)
                )

            elif isinstance(tool,str) and tool in ToolFactory.DEFAULTS:
                tool =ToolFactory.DEFAULTS[tool]

            if tool:

                tools.append(tool)
        return tools
    
    @staticmethod
    def createtool(target,config =None):
        try :

            return CreateTool(target)
        
        except(TypeHintParsingException,TypeError):

            return ToolFactory.fromdocs(target,config if config else {})
        
    @staticmethod
    def fromdocs(target,config):

        name = target.__name__ if isinstance(target,(FunctionType, MethodType)) or not hasattr(target,"__call__") else target.__class__.__name__
        target = target if isinstance(target,(FunctionType,MethodType)) or not hasattr(target,"__call__") else target.__call__

        doc = inspect.getdoc(target)
        description, parameters, _ = chat_template_utils.parse_google_format_docstring(doc.strip()) if doc else (None, {}, None)

        signature = inspect.signature(target)
        inputs = {}
        for pname, param in signature.parameters.items():
            if param.default == inspect.Parameter.empty and pname in parameters:
                inputs[pname] = {"type": "any", "description": parameters[pname]}

        return FunctionTool(
            {
                "name": config.get("name", name.lower()),
                "description": config.get("description", description),
                "inputs": config.get("inputs", inputs),
                "target": config.get("target", target),

            }
        )




