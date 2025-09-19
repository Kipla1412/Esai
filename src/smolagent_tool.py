from src.agent.tool.weather import weather_tool
from smolagents import Tool,ToolCallingAgent,LiteLLMModel

model = LiteLLMModel(
    model_id ="openrouter/meta-llama/llama-3.3-8b-instruct:free",
    api_key ="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",
    api_base ="https://openrouter.ai/api/v1"
 )
agent =ToolCallingAgent(tools =[weather_tool], model =model,add_base_tools =True)
result = agent.run("what is the current weather in madurai?")

print(result)
