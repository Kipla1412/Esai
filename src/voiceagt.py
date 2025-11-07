import yaml
import importlib

from smolagents import Tool
from src.pipeline.llm import LLM
from src.agent.base import Agent
import os
from dotenv import load_dotenv

load_dotenv()

# Your CloudTranscriptionTool class (import from your codebase)
# from src.agent.check.transcrip_tool import CloudTranscriptionTool

# --- Configuration (as dict, can load YAML externally) ---
config = {
    "method": "tool",
    "max_steps": 5,
    "model": {
        "path": "openrouter/meta-llama/llama-3.3-8b-instruct:free",
        "method": "litellm",
        "api_key": os.getenv["HUG_API_KEY"],
        "api_base": "https://openrouter.ai/api/v1",
        "temperature": 0.7
    },
    "tools": [
        {   # Your custom transcription tool config
            "name": "cloud_transcription",
            "description": "Transcribes audio into text using cloud providers (HF/OpenAI/Groq)",
            "target": "src.agent.check.transcrip_tool.CloudTranscriptionTool",
            "params": {
                "endpoint_url": "https://api-inference.huggingface.co/models",
                "api_key": os.getenv["GROQ_API_KEY"],
                "model": "openai/whisper-large-v3",
                "provider": "huggingface"
            }
        }
        # Add other tools here as needed
    ]
}


# Function to dynamically import and instantiate tools from config
def load_tools(tool_configs):
    tools = []
    for tool in tool_configs:
        module_name, class_name = tool["target"].rsplit(".", 1)
        module = importlib.import_module(module_name)
        klass = getattr(module, class_name)
        params = tool.get("params", {})
        tool_instance = klass(**params)
        tools.append(tool_instance)
    return tools


# Load tools
tools = load_tools(config["tools"])


# Initialize LLM
model_conf = config["model"]
llm = LLM(
    model_conf["path"],
    api_key=model_conf.get("api_key"),
    api_base=model_conf.get("api_base"),
    temperature=model_conf.get("temperature"),
    method=model_conf.get("method"),
)

# Create the agent with model, tools, and max_steps
agent = Agent(
    model=llm,
    tools=tools,
    max_iterations=config.get("max_steps", 5),
)

# --- Run agentic multi-step test prompt ---
query = """
Please transcribe the audio file D:/backend/ESAI/src/sample.wav.
and return  the transcription tools output text.
# Please transcribe the audio file D:/backend/ESAI/src/sample.wav and return the transcription text.

# If the transcription is short, like "its okay", consider it valid and return it as is.

# Only respond "Transcription not possible" if the transcription tool fails or returns an error.


"""

response = agent(query)
print("Agent response:\n", response)
