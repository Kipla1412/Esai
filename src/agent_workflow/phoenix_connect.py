import os
import yaml
import logging
from dotenv import load_dotenv
from src.agent.base import Agent
from openai import OpenAI
from phoenix.otel import register
from opentelemetry.trace import Status, StatusCode
import asyncio
# check for evalution

from phoenix.evals import QA_PROMPT_TEMPLATE, QA_PROMPT_RAILS_MAP
from phoenix.evals import  llm_classify
from phoenix.evals.legacy.models.google_genai import GoogleGenAIModel

load_dotenv()

print(os.getenv("OPENROUTER_API_KEY"))
openrouter_key = os.getenv("OPENROUTER_API_KEY")
phoenix_key = os.getenv("PHOENIX_API_KEY")
phoenix_endpoint = os.getenv("PHOENIX_COLLECTOR_ENDPOINT")
phoenix_project = os.getenv("PHOENIX_PROJECT")
gemini_api_key = os.getenv("GEMINI_API_KEY")

if not openrouter_key:
    raise ValueError("OPENROUTER_API_KEY not set in your environment.")
if not phoenix_endpoint:
    raise ValueError("PHOENIX_COLLECTOR_ENDPOINT not set in your environment.")

openai_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=openrouter_key,  # Make sure this is set from env
    default_headers={"HTTP-Referer": "http://localhost"},
)

gemini_model = GoogleGenAIModel(
    model="gemini-2.0-flash",
    api_key=gemini_api_key,
)
print("Phoenix Endpoint:", phoenix_endpoint)
print("Phoenix API Key:", phoenix_key)


tracer_provider = register(
    project_name= phoenix_project,
    #endpoint= phoenix_endpoint,
    api_key= phoenix_key,
    auto_instrument=False, 

)
tracer = tracer_provider.get_tracer(__name__)

async def evaluate_turn_llm(user_input: str, agent_response: str):

    rails =list(QA_PROMPT_RAILS_MAP.values())
    dataframe =[{"input": user_input, "reference": "no medical record", "output": agent_response}]
    
    try:
        results = llm_classify(
            dataframe=dataframe,
            template=QA_PROMPT_TEMPLATE,
            model=gemini_model,
            rails=rails,
            provide_explanation=True,
        )
        if results is None or results.empty:
            return {"classification": "UNKNOWN", "explanation": "No results"}
        # Return the first row as dict
        result = results.iloc[0].to_dict()
        print("Single eval result dict:", result)
        return result
        #return results.iloc[0].to_dict()
    except Exception as e:
        print("[Phoenix Eval Error]", e)
        return {"classification": "error", "explanation": str(e)}


class ConversationalAgent:
    def __init__(self, config_path: str = r"D:\backend\ESAI\src\check.yml"):
        
        self.logger = logging.getLogger("ConversationalAgent")
        self.logger.setLevel(logging.DEBUG)

        with open(config_path) as f:
            config = yaml.safe_load(f)

        self.llm_config = config.get("llm", {})
        self.agent_config = config.get("agent", {})

        full_config = {
            **self.agent_config,
            "model": self.llm_config,
            "tools": []
        }

        self.agent = Agent(**full_config)
        self.conversation_history = ""

        self.initial_message = "Hello! I am your medical assistant. How can I help you today?"

    def reset_conversation(self):
        self.logger.info("Resetting conversation history")
        self.conversation_history = ""

    async def generate_response(self, user_input):
        """
        Generates agent response asynchronously.
        """
        try:
            with tracer.start_as_current_span("generate_agent_response") as span:
                self.conversation_history += f"User: {user_input}\n"

                response = self.agent("conversation_agent", text=self.conversation_history)

                self.conversation_history += f"Agent: {response}\n"
                self.logger.debug(f"Agent response generated: {response}")
                
                
                eval_result = await evaluate_turn_llm(user_input, response)

                #add custom attributes to the phoenix span
                span.set_attribute("input", user_input)
                span.set_attribute("output", response)
                span.set_attribute("user_input", user_input or "N/A")
                span.set_attribute("agent_response", response or "N/A")

                
                classification = eval_result.get("classification") or eval_result.get("label", "unknown")
                explanation = eval_result.get("explanation", "N/A")

                span.set_attribute("phoenix_classification", classification)
                span.set_attribute("phoenix_explanation", explanation)


                return response

        except Exception as e:
            self.logger.error(f"Error during agent response generation: {e}")
            return "Sorry, I encountered an error processing your request."

    def get_initial_message(self):
        return self.initial_message



# import os
# import yaml
# import logging
# from dotenv import load_dotenv
# from src.agent.base import Agent
# from openai import OpenAI
# from phoenix.otel import register
# from opentelemetry.trace import Status, StatusCode
# import asyncio
# # check for evalution

# from phoenix.evals import QA_PROMPT_TEMPLATE, QA_PROMPT_RAILS_MAP
# from phoenix.evals import  llm_classify
# from phoenix.evals.legacy.models.google_genai import GoogleGenAIModel

# load_dotenv()

# print(os.getenv("OPENROUTER_API_KEY"))
# openrouter_key = os.getenv("OPENROUTER_API_KEY")
# phoenix_key = os.getenv("PHOENIX_API_KEY")
# phoenix_endpoint = os.getenv("PHOENIX_COLLECTOR_ENDPOINT")
# phoenix_project = os.getenv("PHOENIX_PROJECT")
# gemini_api_key = os.getenv("GEMINI_API_KEY")

# if not openrouter_key:
#     raise ValueError("OPENROUTER_API_KEY not set in your environment.")
# if not phoenix_endpoint:
#     raise ValueError("PHOENIX_COLLECTOR_ENDPOINT not set in your environment.")

# openai_client = OpenAI(
#     base_url="https://openrouter.ai/api/v1",
#     api_key=openrouter_key,  # Make sure this is set from env
#     default_headers={"HTTP-Referer": "http://localhost"},
# )

# gemini_model = GoogleGenAIModel(
#     model="gemini-2.0-flash",
#     api_key=gemini_api_key,
# )
# print("Phoenix Endpoint:", phoenix_endpoint)
# print("Phoenix API Key:", phoenix_key)


# tracer_provider = register(
#     project_name= phoenix_project,
#     #endpoint= phoenix_endpoint,
#     api_key= phoenix_key,
#     auto_instrument=False, 

# )
# tracer = tracer_provider.get_tracer(__name__)

# async def evaluate_turn_llm(user_input: str, agent_response: str):

#     rails =list(QA_PROMPT_RAILS_MAP.values())
#     dataframe =[{"input": user_input, "reference": "no medical record", "output": agent_response}]
    
#     try:
#         results = llm_classify(
#             dataframe=dataframe,
#             template=QA_PROMPT_TEMPLATE,
#             model=gemini_model,
#             rails=rails,
#             provide_explanation=True,
#         )
#         if results is None or results.empty:
#             return {"classification": "UNKNOWN", "explanation": "No results"}
#         # Return the first row as dict
#         result = results.iloc[0].to_dict()
#         print("Single eval result dict:", result)
#         return result
#         #return results.iloc[0].to_dict()
#     except Exception as e:
#         print("[Phoenix Eval Error]", e)
#         return {"classification": "error", "explanation": str(e)}


# class ConversationalAgent:
#     def __init__(self, config_path: str = r"D:\backend\ESAI\src\check.yml"):
        
#         self.logger = logging.getLogger("ConversationalAgent")
#         self.logger.setLevel(logging.DEBUG)

#         with open(config_path) as f:
#             config = yaml.safe_load(f)

#         self.llm_config = config.get("llm", {})
#         self.agent_config = config.get("agent", {})

#         full_config = {
#             **self.agent_config,
#             "model": self.llm_config,
#             "tools": []
#         }

#         self.agent = Agent(**full_config)
#         self.conversation_history = ""

#         self.initial_message = "Hello! I am your medical assistant. How can I help you today?"

#     def reset_conversation(self):
#         self.logger.info("Resetting conversation history")
#         self.conversation_history = ""

#     async def generate_response(self, user_input):
#         """
#         Generates agent response asynchronously.
#         """
#         try:
#             with tracer.start_as_current_span("generate_agent_response") as span:
#                 self.conversation_history += f"User: {user_input}\n"

#                 response = self.agent("conversation_agent", text=self.conversation_history)

#                 self.conversation_history += f"Agent: {response}\n"
#                 self.logger.debug(f"Agent response generated: {response}")
                
                
#                 eval_result = await evaluate_turn_llm(user_input, response)

#                 #add custom attributes to the phoenix span
#                 span.set_attribute("input", user_input)
#                 span.set_attribute("output", response)
#                 span.set_attribute("user_input", user_input or "N/A")
#                 span.set_attribute("agent_response", response or "N/A")

                
#                 classification = eval_result.get("classification") or eval_result.get("label", "unknown")
#                 explanation = eval_result.get("explanation", "N/A")

#                 span.set_attribute("phoenix_classification", classification)
#                 span.set_attribute("phoenix_explanation", explanation)


#                 return response

#         except Exception as e:
#             self.logger.error(f"Error during agent response generation: {e}")
#             return "Sorry, I encountered an error processing your request."

#     def get_initial_message(self):
#         return self.initial_message
