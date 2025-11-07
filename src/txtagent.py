import yaml
import importlib
from src.agent.base import Agent


with open(r"D:\backend\ESAI\src\config.yaml") as f:
    config = yaml.safe_load(f)

for tool in config["tools"]:
    if "target" in tool and isinstance(tool["target"], str):
        module_name, func_name = tool["target"].rsplit(".", 1)
        module = importlib.import_module(module_name)
        #tool_class = getattr(module, class_name)
        #target_obj =getattr(module, func_name)
        tool["target"] = getattr(module, func_name)
        #tool["target"] = getattr(module, func_name)


agent = Agent(**config)

#aprint(agent("What is the most expensive product in the Electronics category?"))
print(agent("Transcribe the file D:/backend/sample.wav"))