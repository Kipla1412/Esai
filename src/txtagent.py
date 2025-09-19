import yaml
import importlib
from src.agent.base import Agent


with open(r"D:\backend\ESAI\src\config.yaml") as f:
    config = yaml.safe_load(f)

for tool in config["tools"]:
    if "target" in tool and isinstance(tool["target"], str):
        module_name, func_name = tool["target"].rsplit(".", 1)
        module = importlib.import_module(module_name)
        tool["target"] = getattr(module, func_name)


agent = Agent(**config)

print(agent("what is current weather in Chennai? and What is 2 plus 3 times 4?"))
