# import numpy as np
# import io
# import soundfile as sf
# import asyncio
# from fastapi import FastAPI, WebSocket, WebSocketDisconnect
# from starlette.websockets import WebSocketState
# # Assuming these imports are correct based on your file structure
# from src.agent_workflow.agent_check import ConversationalAgent
# from src.pipeline.audio.speech_to_text import SpeechToText
# from src.pipeline.audio.cloud_tts import CloudTextToSpeech

# app = FastAPI()

# chat_agent = ConversationalAgent()
# stt = SpeechToText(provider="hf-inference", api_key="")
# tts = CloudTextToSpeech(
#     endpoint_url="https://api.groq.com/openai/v1", 
#     api_key="", 
#     model="playai-tts",
#     provider="groq",
#     voice="Aaliyah-PlayAI" 
# )

# # --- Reliable Sender Task (No change needed, it's correct) ---
# async def sender_task(websocket: WebSocket, queue: asyncio.Queue):
#     """Background task to reliably send messages from the queue one by one."""
#     while True:
#         try:
#             message = await queue.get()
            
#             type = message.get("type")
#             data = message.get("data")
            
#             if type == "text":
#                 await websocket.send_text(data)
#             elif type == "bytes":
#                 await websocket.send_bytes(data)
            
#             queue.task_done()
            
#         except WebSocketDisconnect:
#             break
#         except Exception as e:
#             print(f"Error in sender task: {e}")
#             break

# # ----------------------------------


# @app.websocket("/chat")
# async def websocket_chat(ws: WebSocket):
#     await ws.accept()

#     conversation_history = ""
#     audio_buffer = []  # Stores raw 16-bit PCM byte chunks
#     is_recording = False
#     sample_rate = 16000 # Standard for client-side mic recording

#     # 1. Setup Queue and Start Sender Task
#     send_queue = asyncio.Queue()
#     sender = asyncio.create_task(sender_task(ws, send_queue)) 

#     # --- Initial Greeting ---
#     initial_text = chat_agent.get_initial_message()
#     conversation_history = f"Agent: {initial_text}\n"

#     # Enqueue text
#     await send_queue.put({"type": "text", "data": f"Agent: {initial_text}"})
    
#     try:
#         audio_np, sr = await asyncio.to_thread(tts, initial_text)
#         # --- REFINEMENT: Send PCM bytes for consistency ---
#         pcm_bytes = tts.float_to_pcm16(audio_np)
#         await send_queue.put({"type": "bytes", "data": pcm_bytes})
#     except Exception as e:
#         print(f"Initial TTS failed: {e}")

#     # --- Main Conversation Loop ---
#     try:
#         while True:
#             msg = await ws.receive()

#             if msg["type"] != "websocket.receive":
#                 continue

#             # ----- TEXT/COMMAND message from user -----
#             if "text" in msg:
#                 command = msg["text"].strip().lower()
                
#                 if command == "start_audio":
#                     is_recording = True
#                     audio_buffer = []  
#                     await send_queue.put({"type": "text", "data": "SYSTEM: Recording started."})

#                 elif command == "stop_audio":
#                     is_recording = False
#                     await send_queue.put({"type": "text", "data": "SYSTEM: Processing audio..."})
                    
#                     if not audio_buffer:
#                         await send_queue.put({"type": "text", "data": "SYSTEM: No audio recorded."})
#                         continue
                    
#                     # 1. Combine all raw 16-bit PCM chunks
#                     full_audio_bytes = b"".join(audio_buffer)
                    
#                     # 2. Convert 16-bit PCM bytes to float32 NumPy array (ready for STT pipeline)
#                     audio_np = np.frombuffer(full_audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

#                     if audio_np.size == 0:
#                         continue
#                     buffer = io.BytesIO()
#                     # Ensure the audio is mono (if necessary, though the client should provide mono)
#                     if audio_np.ndim > 1:
#                         audio_np = audio_np.mean(axis=1) 
                        
#                     sf.write(buffer, audio_np, sample_rate, format="WAV")
#                     wav_bytes = buffer.getvalue()
                    
#                     try:
#                         # Use asyncio.to_thread for the synchronous network request
#                         user_text = await asyncio.to_thread(
#                             stt,
#                             audio_np,
#                             sample_rate, 
#                             model="openai/whisper-large-v3" # or "facebook/wav2vec2-large-960h-lv60-self" for speed
#                         )
#                         # user_text = result.get("text", "").strip()
                        
#                     except Exception as e:
#                         print(f"[STT API ERROR] STT Failed: {e}")
#                         await send_queue.put({"type": "text", "data": "SYSTEM: Transcription error or timeout."})
#                         continue    
#                     # --- STT for the accumulated audio (This line uses your STT class) ---
#                     # try:
#                     #     user_text = await stt(audio_np, sample_rate, model="openai/whisper-large-v3")
#                     # except Exception as e:
#                     #     print(f"[STT ERROR] STT Failed: {e}")
#                     #     await send_queue.put({"type": "text", "data": "SYSTEM: Transcription error."})
#                     #     continue
                        
#                     if not user_text.strip():
#                         await send_queue.put({"type": "text", "data": "SYSTEM: Could not recognize speech."})
#                         continue
                        
#                     # --- Agent & TTS Response ---
#                     conversation_history += f"User: {user_text}\n"
#                     await send_queue.put({"type": "text", "data": f"User (spoken): {user_text}"})
                    
#                     response = await asyncio.to_thread(chat_agent.agent, "conversation_agent", text=conversation_history)
#                     conversation_history += f"Agent: {response}\n"

#                     await send_queue.put({"type": "text", "data": f"Agent: {response}"})
                    
#                     try:
#                         audio_np, sr = await asyncio.to_thread(tts, response)
#                         pcm_bytes = tts.float_to_pcm16(audio_np)
#                         await send_queue.put({"type": "bytes", "data": pcm_bytes})
#                     except Exception as e:
#                         print("TTS error:", e)

#                 # Handle direct text input
#                 else:
#                     user_text = command
#                     if user_text:
#                         conversation_history += f"User: {user_text}\n"
#                         await send_queue.put({"type": "text", "data": f"User: {user_text}"})
                        
#                         response = await asyncio.to_thread(chat_agent.agent, "conversation_agent", text=conversation_history)
#                         conversation_history += f"Agent: {response}\n"
#                         await send_queue.put({"type": "text", "data": f"Agent: {response}"})

#                         try:
#                             audio_np, sr = await asyncio.to_thread(tts, response)
#                             pcm_bytes = tts.float_to_pcm16(audio_np)
#                             await send_queue.put({"type": "bytes", "data": pcm_bytes})
#                         except Exception as e:
#                             print("TTS error:", e)


#             # ----- AUDIO message from user (chunks) -----
#             elif "bytes" in msg and is_recording:
#                 # Accumulate raw 16-bit PCM audio chunks (no processing here!)
#                 audio_buffer.append(msg["bytes"])

#     except WebSocketDisconnect:
#         print("Client disconnected")
#     except Exception as e:
#         print("WebSocket error:", e)
    
#     finally:
#         sender.cancel() 
#         if ws.client_state != WebSocketState.DISCONNECTED:
#             await ws.close()
#         print("WebSocket closed")
# main.py or a separate script like dashboard_start.py
# main.py or a separate script like dashboard_start.py
# from trulens_eval import Tru

# # Initialize the TruLens handler
# # This object manages the connection to your evaluation database (default.sqlite)
# tru = Tru()

# # Launch the dashboard on the specified port
# # Note: Tru.run_dashboard() blocks the execution of the script.
# print("Starting TruLens dashboard on http://localhost:8000 ...")
# tru.run_dashboard(port=8000)

# from dotenv import load_dotenv
# import os

# load_dotenv()
# print("OPENROUTER_API_KEY:", os.getenv("OPENROUTER_API_KEY"))
# 
 
# from fastapi import FastAPI
# from dotenv import load_dotenv
# import os
# load_dotenv(dotenv_path=r"D:\backend\ESAI\src\.env")  # full path

# print("OPENROUTER_API_KEY:", os.getenv("OPENROUTER_API_KEY"))  # Should print your key

# app = FastAPI()

# @app.get("/")
# async def root():
#     return {"message": "Hello World"}
"""
import phoenix.evals
print(dir(phoenix.evals))

print("*****************************************************************")
import phoenix.evals.templates
print(dir(phoenix.evals.templates))
print("*****************************************************************")
from phoenix.evals import GoogleGenAIModel
help(GoogleGenAIModel.__init__)
print("*****************************************************************")
import phoenix.evals.legacy.models.google_genai as gg
print(dir(gg))

help(GoogleGenAIModel)

"""
# from phoenix.evals import QA_PROMPT_TEMPLATE


# print("Template object:", QA_PROMPT_TEMPLATE)
# print("Expected template variables:", QA_PROMPT_TEMPLATE.variables)
# print("Template fields:", QA_PROMPT_TEMPLATE.__dict__.keys())
# """
# python -m src.check
# (myenv) PS D:\backend\ESAI> python -m src.check
# Template object: 
# You are given a question, an answer and reference text. You must determine whether the
# given answer correctly answers the question based on the reference text. Here is the data:       
#     [BEGIN DATA]
#     ************
#     [Question]: {input}
#     ************
#     [Reference]: {reference}
#     ************
#     [Answer]: {output}
#     [END DATA]
# Your response must be a single word, either "correct" or "incorrect",
# and should not contain any text or characters aside from that word.
# "correct" means that the question is correctly and fully answered by the answer.
# "incorrect" means that the question is not correctly or only partially answered by the
# answer.

# Expected template variables: ['input', 'reference', 'output']
# Template fields: dict_keys(['rails', 'template', 'explanation_template', 'explanation_label_parse
# r', '_start_delim', '_end_delim', 'variables', '_scores'])                                       
# """
import mlflow
import google.genai as genai
import os
from dotenv import load_dotenv

load_dotenv()
mlflow.gemini.autolog()

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Gemini")

client = genai.Client(api_key= os.getenv("GEMINI_API_KEY"))

response = client.models.generate_content(model="gemini-2.5-flash", contents="what is Ai?")
print(response)