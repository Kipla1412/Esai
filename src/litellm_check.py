
from src.pipeline.llm.litellm import LiteLLM
import os

# os.environ["GROQ_API_KEY"] =""

# llm =LiteLLM(path ="groq/llama3-8b-8192")

# prompt = "tell me about moon"

# messages =[{"role":"user", "content":prompt}]

# result = llm.stream([messages],maxlength=250,stream= False,stop=None)

# response =next(result)

# print("Response:",response)

API_KEY = ""

MODEL_PATH = "openrouter/meta-llama/llama-3.3-8b-instruct:free"

llm = LiteLLM(
    MODEL_PATH,
    api_key=API_KEY,
    api_base="https://openrouter.ai/api/v1"
    )
prompt =input("Enter ur query :")

messages = [{"role":"user", "content":prompt}]


results =llm.stream([messages],maxlength=250,stream=True,stop =None)
full_response = ""

for chunk in results:
    try:
        text = chunk["choices"][0]["message"]["content"]
    except Exception:
        text = str(chunk)

    print(text, end="", flush=True)   # live streaming
    full_response += text

print(" Final Response:")
print(full_response)

# response = next(results)
# print (response)