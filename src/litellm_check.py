
from src.pipeline.llm.litellm import LiteLLM
import os

os.environ["GROQ_API_KEY"] ="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"

llm =LiteLLM(path ="groq/llama3-8b-8192")

prompt = "tell me about moon"

messages =[{"role":"user", "content":prompt}]

result = llm.stream([messages],maxlength=250,stream= False,stop=None)

response =next(result)

print("Response:",response)