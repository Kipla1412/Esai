import yaml
import logging
from src.agent.base import Agent
import mlflow
from mlflow.entities import SpanType
import asyncio


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

        #log statatic run parameters

    def reset_conversation(self):
        self.logger.info("Resetting conversation history")
        self.conversation_history = ""

    @mlflow.trace(name="generate_response", span_type=SpanType.AGENT)
    async def generate_response(self, user_input):

        """
        Generates agent response asynchronously.
        """
        try:
            self.conversation_history += f"User: {user_input}\n"

            # Assume agent() is synchronous; can be called inside asyncio.to_thread if needed
            response = self.agent("conversation_agent", text=self.conversation_history)

            self.conversation_history += f"Agent: {response}\n"
            self.logger.debug(f"Agent response generated: {response}")
        
            mlflow.set_tag("request", user_input)
            mlflow.set_tag("response", response)
            # # add mlflow logging here

            # mlflow.log_metric("latest_user_input", len(user_input))
            # mlflow.log_metric("latest_agent_output", len(response))
            # mlflow.log_metric("conversation_length", len(self.conversation_history))

            # turn_log = f"User: {user_input}\nAgent: {response}\n"
            # filename = f"turn_{int(time.time())}.txt"
            # mlflow.log_text(turn_log, filename)

            return response

        except Exception as e:
            self.logger.error(f"Error during agent response generation: {e}")
            #mlflow.log_param("agent_error", str(e))
            return "Sorry, I encountered an error processing your request."

    def get_initial_message(self):
        return self.initial_message
    
    @staticmethod
    def log_static_params(agent):
        mlflow.log_param("llm_method", agent.llm_config.get("method"))
        mlflow.log_param("llm_model", agent.llm_config.get("path"))
        mlflow.log_param("llm_temperature", agent.llm_config.get("temperature"))
        mlflow.log_param("agent_description", agent.agent_config.get("description"))
        
        max_iter = agent.agent_config.get("max_iterations")
        if max_iter is not None:
            mlflow.log_param("max_iterations", max_iter)
"""
if __name__ == "__main__":

    if mlflow.active_run() is not None:
        mlflow.end_run()

    agent_instance = ConversationalAgent()
    conversation_history = ""
    print("Agent is ready. Starting terminal chat:")

    with mlflow.start_run(run_name="terminal_session"):
   
        agent_instance.log_static_params(agent_instance)

        while True:
            try:
                # Get user input first
                user_input = input("You: ")
                # Call async traced generate_response method
                question = asyncio.run(agent_instance.generate_response(user_input))
                print("Agent:", question)

            except Exception as e:
                print("Error during agent processing:", e)
                break
    
"""






    # with mlflow.start_run(run_name ="terminal_session"):
    #     while True:
    #         try:
    #             # Get agent output based on conversation so far
    #             question = agent_instance.agent("conversation_agent", text=conversation_history)
    #             print("Agent:", question)

    #             # Get user input
    #             answer = input("You: ")

    #             # Append to conversation history for context
    #             conversation_history += f"Agent: {question}\nUser: {answer}\n"

    #         except Exception as e:
    #             print("Error during agent processing:", e)
    #             break


"""phoenix integrations"""
# import os
# import yaml
# import logging
# from dotenv import load_dotenv
# from src.agent.base import Agent
# from openai import OpenAI
# from phoenix.otel import register
# # from opentelemetry.sdk.trace.export import BatchSpanProcessor
# # from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
# from opentelemetry.trace import Status, StatusCode
# # check for evalution

# import phoenix.evals.templates.default_templates as templates

# load_dotenv()

# print(os.getenv("OPENROUTER_API_KEY"))
# openrouter_key = os.getenv("OPENROUTER_API_KEY")
# phoenix_key = os.getenv("PHOENIX_API_KEY")
# phoenix_endpoint = os.getenv("PHOENIX_COLLECTOR_ENDPOINT")
# phoenix_project = os.getenv("PHOENIX_PROJECT")

# if not openrouter_key:
#     raise ValueError("OPENROUTER_API_KEY not set in your environment.")
# if not phoenix_endpoint:
#     raise ValueError("PHOENIX_COLLECTOR_ENDPOINT not set in your environment.")

# openai_client = OpenAI(
#     base_url="https://openrouter.ai/api/v1",
#     api_key=openrouter_key,  # Make sure this is set from env
#     default_headers={"HTTP-Referer": "http://localhost"},
# )
# print("Phoenix Endpoint:", phoenix_endpoint)
# print("Phoenix API Key:", phoenix_key)


# tracer_provider = register(
#     project_name= phoenix_project,
#     #collector_endpoint= phoenix_endpoint,
#     api_key= None,
#     auto_instrument=False, 

# )
# tracer = tracer_provider.get_tracer(__name__)


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

#                 #add custom attributes to the phoenix span

#                 span.set_attribute("user_input", user_input)
#                 span.set_attribute("agent_response", response)

#                 return response

#         except Exception as e:
#             self.logger.error(f"Error during agent response generation: {e}")
#             return "Sorry, I encountered an error processing your request."

#     def get_initial_message(self):
#         return self.initial_message



# if __name__ == "__main__":
#     import asyncio

#     async def main():
#         agent = ConversationalAgent()
#         print(agent.get_initial_message())
#         response = await agent.generate_response("What are the symptoms of dengue?")
#         print("Agent:", response)

#     asyncio.run(main())


# import yaml
# import asyncio
# import json
# from src.agent.base import Agent

# class Chat_Assistant:
#     def __init__(self, config_path: str = r"D:\backend\ESAI\src\check.yml"):
#         with open(config_path) as f:
#             config = yaml.safe_load(f)

#         self.llm_config = config.get("llm", {})
#         self.agent_config = config.get("agent", {})

#         full_config = {
#             **self.agent_config,
#             "model": self.llm_config,
#             "tools": []
#         }

#         self.chat_agent = Agent(**full_config)
#         print("Agent initialized successfully")

#     async def chat(self, conversation_history: str) -> str:
#         trimmed_history = self.trim_history(conversation_history)
#         last_user_message = self.get_last_user_message(conversation_history)

#         prompt_template = self.agent_config["prompt_templates"]["system_prompt"]
#         formatted_prompt = prompt_template.format(
#             text=trimmed_history,
#             last_user_message=last_user_message
#         )

#         for attempt in range(2):
#             try:
#                 print(f"Attempt {attempt + 1}: sending prompt")
#                 print("Prompt:", repr(formatted_prompt))
#                 raw_response = await asyncio.to_thread(
#                     self.chat_agent, "conversation_agent", formatted_prompt
#                 )
#                 print("Raw agent response:", repr(raw_response))

#                 if not raw_response or raw_response.strip() == "":
#                     print("Empty response from model, retrying...")
#                     continue

#                 cleaned_response = raw_response.strip()

#                 if not cleaned_response.startswith("{"):
#                     start_index = cleaned_response.find("{")
#                     if start_index != -1:
#                         cleaned_response = cleaned_response[start_index:]
#                     else:
#                         print("⚠ No valid JSON found in response.")
#                         return "Sorry, I didn't understand that. Could you please rephrase?"

#                 try:
#                     data = json.loads(cleaned_response)
#                 except json.JSONDecodeError as e:
#                     print(f"JSON decode error: {e} | Raw: {raw_response}")
#                     continue

#                 if "arguments" in data and "answer" in data["arguments"]:
#                     return data["arguments"]["answer"]
#                 else:
#                     print("Missing 'answer' key in JSON:", data)
#                     return "Sorry, I didn't understand that. Could you please rephrase?"

#             except Exception as e:
#                 print(f"Attempt {attempt + 1} failed with error: {e}")

#         return "Sorry, I didn't understand that. Could you please rephrase?"

#     @staticmethod
#     def trim_history(conversation_history: str, max_turns: int = 4) -> str:
#         if not conversation_history:
#             return ""
#         lines = [line.strip() for line in conversation_history.split("\n") if line.strip()]
#         if len(lines) > max_turns * 2:
#             lines = lines[-max_turns * 2:]
#         return "\n".join(lines)

#     @staticmethod
#     def get_last_user_message(conversation_history: str) -> str:
#         lines = conversation_history.strip().split("\n")
#         for line in reversed(lines):
#             if line.startswith("User:"):
#                 return line.replace("User:", "").strip()
#         return ""

#     async def first_message(self) -> tuple[str, str]:
#         greeting_text = "Hello! I'm your friendly medical assistant. How can I help you today?"
#         conversation_history = f"Agent: {greeting_text}\n"
#         return greeting_text, conversation_history

#     @staticmethod
#     def extract_response(raw_response: str) -> str:
#         try:
#             cleaned_response = raw_response.strip()
#             if not cleaned_response.startswith("{"):
#                 start_index = cleaned_response.find("{")
#                 if start_index != -1:
#                     cleaned_response = cleaned_response[start_index:]

#             data = json.loads(cleaned_response)
#             if "arguments" in data:
#                 return data["arguments"].get("answer") or next(iter(data["arguments"].values()), "")
#             elif "final_answer" in data:
#                 return data["final_answer"]
#             return cleaned_response

#         except Exception as e:
#             print(f"Error parsing JSON response: {e}")
#             return "Sorry, I didn't understand that. Could you please rephrase?"

# import yaml
# from src.agent.base import Agent
# import asyncio
# import json
# import traceback

# class Chat_Assistant:

#     def __init__(self, config_path:str =r"D:\backend\ESAI\src\check.yml"):

#         with open(config_path) as f:

#             config = yaml.safe_load(f)

#         llm_config = config.get("llm", {})
#         agent_config = config.get("agent", {})

#         full_config = {
#             **agent_config,
#             "model": llm_config,
#             "tools": []
#         }

#         self.chat_agent =Agent(**full_config)
#         self.agent_config = agent_config
#         print("Agent initialized successfully")

#     async def chat(self, conversation_history: str) -> str:

#         trimmed_history = self.trim_history(conversation_history)

#         last_user_message = self.get_last_user_message(conversation_history)
#         prompt_template = self.agent_config["prompt_templates"]["system_prompt"]
#         formatted_prompt = prompt_template.format(
#             text=trimmed_history,
#             last_user_message=last_user_message
#         )
        
#         for attempt in range(2):
#             try:
#                 print(f"Attempt {attempt + 1}: sending conversation history to agent")
#                 print(conversation_history)
#                 raw_response = await asyncio.to_thread(self.chat_agent, "conversation_agent", formatted_prompt)
#                 print("Raw agent response:", repr(raw_response))

#                 if not raw_response.strip():
#                     print("Empty raw response, retrying...")
#                     continue
#                     #print("Raw agent response:", raw_response)
                
#                 try:
#                     data = json.loads(raw_response)
#                     if "arguments" in data and "answer" in data["arguments"]:
#                         return data["arguments"]["answer"]
#                     else:
#                         raise ValueError("Missing 'answer' key,data:", data)
#                 except json.JSONDecodeError:
#                     print(f"JSON decode error: {e}")
#                     print("Raw response that caused error:", repr(raw_response))
#                     # If response is not valid JSON, return raw text directly
#                     return raw_response

#             except Exception as e:
#                 print(f"Attempt {attempt + 1} failed with error: {e}")
#                 print(traceback.format_exc())
#         return "Sorry, I didn't understand that. Could you please rephrase?"
    
#     def trim_history(self, conversation_history: str, max_turns: int = 4) -> str:
#         """
#         Keeps only the last few turns of the conversation
#         to avoid long context and repeated greetings.
#         """
#         if not conversation_history:
#             return ""

#         lines = [line.strip() for line in conversation_history.split("\n") if line.strip()]

#         # Keep only last N user-agent pairs
#         if len(lines) > max_turns * 2:
#             lines = lines[-max_turns * 2:]

#         return "\n".join(lines)
    
#     def get_last_user_message(self, conversation_history: str) -> str:
#         """
#         Extracts the latest user message from the conversation history.
#         """
#         lines = conversation_history.strip().split("\n")
#         for line in reversed(lines):
#             if line.startswith("User:"):
#                 return line.replace("User:", "").strip()
#         return ""
        
#     async def first_message(self) -> tuple[str, str]:
#         """
#         Send the agent's first message automatically.
#         """
#         greeting_text = "Hello! I'm your friendly medical assistant. How can I help you today?"
#         conversation_history = f"Agent: {greeting_text}\n"
#         return greeting_text, conversation_history
    
#     @staticmethod
#     def extract_response(raw_response: str) -> str:
#         """Safely extract 'answer' from LLM JSON output."""
#         try:
#             clean_response = raw_response.strip()
#             data = json.loads(clean_response)
#             if "arguments" in data:
#                 if "answer" in data["arguments"]:
#                     return data["arguments"]["answer"]
                
#                 if "message" in data["arguments"]:
#                     return data["arguments"]["message"]
                
#                 values = list(data["arguments"].values())
#                 if values:
#                     return values[0]
                
#             if "final_answer" in data:
#                 return data["final_answer"]
            
#             return clean_response
#         except json.JSONDecodeError:
#             print("Warning: JSON decode error, returning raw response without parsing.")
#             return raw_response
        
#         except Exception as e:
#             print(f"Error parsing JSON response: {e}")
#             return raw_response 

            
# if __name__ == "__main__":
#     conversation_history = ""

#     async def main():
#         print("Agent ready! Type something to start chatting.")

#         # Show agent’s first message automatically
#         assistant = Chat_Assistant()
#         greeting, conversation_history = await assistant.first_message()
#         print("Agent:", greeting)

#         # Continuous chat loop
#         while True:
#             user_input = input("You: ")
#             conversation_history += f"User: {user_input}\n"
#             print("Conversation before agent call:\n", conversation_history)
#             reply = await assistant.chat(conversation_history)
#             print("Agent:", reply)
#             conversation_history += f"Agent: {reply}\n"
#             print("Conversation after agent reply:\n", conversation_history)

#     asyncio.run(main())
    
#     # async def chat(self, conversation_history : str) -> str:

#     #     """
#     #     Runs the agent with given conversation history.
#     #     Returns the agent response.
#     #     """
#     #     try:
#     #         raw_response = await asyncio.to_thread(self.chat_agent, "conversation_agent", conversation_history)
#     #         return self.extract_response(raw_response)
            
#     #         # try:
#     #         #     data =json.loads(raw_response)
#     #         #     reply = data["arguments"]["response"]

#     #         # except Exception:
#     #         #     reply =raw_response
#     #         #return reply
#     #     except Exception as e:
#     #         return f"Error during agent processing: {e}"