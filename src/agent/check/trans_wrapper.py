# from src.agent.check.transcrip_tool import CloudTranscriptionTool

# class CloudTranscriptionWrapper:
#     """
#     Wrapper for CloudTranscriptionTool to be compatible with txtai YAML + ToolFactory.
#     Ignores extra kwargs like 'target' and passes only the needed parameters.
#     """

#     def __init__(self, **kwargs):
#         # Extract only what CloudTranscriptionTool expects
#         tool_params = {k: kwargs[k] for k in ["endpoint_url", "api_key", "model", "provider"] if k in kwargs}
#         print("CloudTranscriptionWrapper init kwargs:", kwargs)

#         self.tool = CloudTranscriptionTool(**tool_params)

#     def __call__(self, *args, **kwargs):
#         # Forward call to the actual tool
#         if args:
#             return self.tool(args[0])
#         elif "audio_path" in kwargs:
#             return self.tool(kwargs["audio_path"])
#         else:
#             raise ValueError("Missing 'audio_path'")


import yaml
from src.agent.check.transcrip_tool import CloudTranscriptionTool

class CloudTranscriptionWrapper:
    """
    Wrapper for CloudTranscriptionTool compatible with Smolagents.
    - Uses kwargs if passed
    - Falls back to reading params from YAML if kwargs is empty
    """

    def __init__(self, **kwargs):
        if kwargs:
            tool_params = {k: kwargs[k] for k in ["endpoint_url", "api_key", "model", "provider"] if k in kwargs}
        else:
            # Fallback: load from YAML file directly
            try:
                CONFIG_PATH = "D:/backend/ESAI/src/config1.yaml"
                with open(CONFIG_PATH, "r") as f:
                    config = yaml.safe_load(f)
                tool_config = config["transcription_agent"]["tools"][0]
                tool_params = tool_config.get("params", {})
            except Exception as e:
                raise RuntimeError(f"Failed to load tool params from YAML: {e}")

        print("CloudTranscriptionWrapper init kwargs:", tool_params)

        if "endpoint_url" not in tool_params:
            raise ValueError("endpoint_url is required for CloudTranscriptionTool")

        self.tool = CloudTranscriptionTool(**tool_params)

    def __call__(self, *args, **kwargs):
        if args:
            return self.tool(args[0])
        elif "audio_path" in kwargs:
            return self.tool(kwargs["audio_path"])
        else:
            raise ValueError("Missing 'audio_path'")
