import yaml
import importlib
from src.agent.tool.factory import ToolFactory

# Load config
with open(r"D:\backend\ESAI\src\config.yaml") as f:
    config = yaml.safe_load(f)


for tool in config["tools"]:
    if "target" in tool and isinstance(tool["target"], str):
        module_name, func_name = tool["target"].rsplit(".", 1)
        module = importlib.import_module(module_name)
        tool["target"] = getattr(module, func_name)


tools = ToolFactory.create(config)


for t in tools:
    print(t.name, t.description, getattr(t, "inputs", None))


print("Add result:", tools[0].forward(5,7))
print("multiply result:",tools[1].forward(7,9))
