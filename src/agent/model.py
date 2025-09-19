import re
from enum import Enum

from smolagents import ChatMessage,Model,get_clean_message_list,tool_role_conversions
from smolagents.models import get_tool_call_from_text, remove_stop_sequences

from ..pipeline import LLM

class PipelineModel(Model):

    def __init__(self,path =None,method =None,**kwargs):

        self.llm = path if isinstance(path,LLM) else LLM(path,method,**kwargs)
        self.maxlength =8192

        self.model_id = self.llm.generator.path

        super().__init__(flatten_messages_as_text = not self.llm.isvision(), **kwargs)
        
    def generate(self,messages,stop_sequences= None,response_format =None,tools_to_call_from =None,**kwargs):

        messages =self.clean(messages) 
        response = self.llm(messages,maxlength =self.maxlength,stop=stop_sequences,**kwargs)

        # if "Thought:" in response and "Action:" in response:
           
        #    thought_part,actions_part = response.split("Action:",1)
        #    print("\n━━━━━━━━━━ Thought ━━━━━━━━━━")
        #    print(thought_part)
        #    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

        #    print("\n━━━━━━━━━━ Action ━━━━━━━━━━")
        #    print("Action:" + actions_part.strip())
        #    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
        # else:
        #    print("\n[LLM OUTPUT RAW]\n", response)

        if stop_sequences is not None:
            response =remove_stop_sequences(response,stop_sequences)

        message =ChatMessage(role ="assistant", content=response)

        if tools_to_call_from:
            message.tool_calls =[

                get_tool_call_from_text(
                    re.sub(r".*?Action:(.*?\n\}).*", r"\1", response, flags=re.DOTALL), self.tool_name_key, self.tool_arguments_key
                )
            ]
        return message
    
    def parameters(self,maxlength):

        self.maxlength = maxlength


    def clean(self,messages):

        messages = get_clean_message_list(messages,role_conversions =tool_role_conversions,flatten_messages_as_text =self.flatten_messages_as_text)

        for message in messages:

            if "role" in message:

                message["role"] = message["role"].value if isinstance(message["role"],Enum) else message["role"]

        return messages 

