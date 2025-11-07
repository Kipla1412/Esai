# #from dotenv import load_dotenv

# import numpy as np
# import sounddevice as sd
# from  src.pipeline.llm.litellm import LiteLLM

# model_path ="openrouter/google/gemini-2.5-flash-lite-preview-06-17"

# llm = LiteLLM(model_path, 
#               api_key="", 
#               api_base="https://openrouter.ai/api/v1"
#               )
# audio_check = llm.text_to_speech("Hello Kipla, What's up.", voice ="default")

# audio_data =np.frombuffer(audio_check, dtype=np.int16)

# sd.play(audio_data, samplerate=22050)
# sd.wait()

import numpy as np
import os
from src.pipeline.llm.litellm import LiteLLM

os.environ["OPENAI_API_KEY"] = ""
os.environ["LITELLM_BASE_URL"] = "https://api.groq.com/openai/v1"
model_path ="groq/playai-tts"

#speech to text model
# model_path ="groq/whisper-large-v3-turbo"
llm = LiteLLM(model_path)


# audio_bytes = llm.text_to_speech("Hello Kipla what's up", voice="Arista-PlayAI")
# llm.play_audio(audio_bytes)

while True:
    text = input("Enter your text (or 'quit' to exit): ")

    if text.lower() in ["quit", "exit"]:
        break

    audio_bytes =llm.text_to_speech(text, voice= "Arista-PlayAI")
    llm.play_audio(audio_bytes)

