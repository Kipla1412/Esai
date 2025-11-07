import yaml
import logging
from src.agent.base import Agent

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
            self.conversation_history += f"User: {user_input}\n"

            # Assume agent() is synchronous; can be called inside asyncio.to_thread if needed
            response = self.agent("conversation_agent", text=self.conversation_history)

            self.conversation_history += f"Agent: {response}\n"
            self.logger.debug(f"Agent response generated: {response}")
            return response

        except Exception as e:
            self.logger.error(f"Error during agent response generation: {e}")
            return "Sorry, I encountered an error processing your request."

    def get_initial_message(self):
        return self.initial_message


if __name__ == "__main__":
    agent_instance = ConversationalAgent()
    conversation_history = ""
    print("Agent is ready. Starting terminal chat:")

    while True:
        try:
            # Get agent output based on conversation so far
            question = agent_instance.agent("conversation_agent", text=conversation_history)
            print("Agent:", question)

            # Get user input
            answer = input("You: ")

            # Append to conversation history for context
            conversation_history += f"Agent: {question}\nUser: {answer}\n"

        except Exception as e:
            print("Error during agent processing:", e)
            break



# from src.agent.base import Agent
# import yaml

# with open(r"D:\backend\ESAI\src\check.yml") as f:
#     config = yaml.safe_load(f)
   
# llm_config = config.get("llm", {})
# agent_config = config.get("agent", {})

# full_config = {
#     **agent_config,
#     "model": llm_config,
#     "tools": []    # model should be nested here
# }
# #agent = Agent(model=llm_config, **agent_config)
# agent =Agent(**full_config)

# conversation_history =""

# print ("agent is worked")

# while True:
#     try:
#         question = agent("conversation_agent", conversation_history)
#         print("Agent:", question)

#         answer = input("You: ")
#         conversation_history += f"Agent: {question}\nUser: {answer}\n"

#     except Exception as e:
#         print("Error during agent processing:", e)
#         break