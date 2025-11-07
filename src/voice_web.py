from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState
import numpy as np
import asyncio
from src.agent_workflow.agent import ConversationalAgent
from src.pipeline.audio.transcription import CloudTranscription
from src.pipeline.audio.cloud_tts import CloudTextToSpeech
from fastapi.middleware.cors import CORSMiddleware
import mlflow
from mlflow.entities import SpanType
import time
import os
import traceback
from google import genai
from dotenv import load_dotenv

mlflow.set_experiment("voice_agent")
mlflow.dspy.autolog()


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

chat_agent = ConversationalAgent()

stt = CloudTranscription(
    endpoint_url= "https://api-inference.huggingface.co/models/openai/whisper-large", #"https://api-inference.huggingface.co",
    api_key= os.getenv["HUG_API_KEY"],
    model="openai/whisper-large",
    provider="huggingface",
    chunk=10,
    target_rate=16000
)
tts = CloudTextToSpeech(
    endpoint_url="https://api.groq.com/openai/v1",
    api_key=os.getenv["GROQ_API_KEY"],
    model="playai-tts",
    provider="groq",
    voice="Arista-PlayAI"
)


# CONFIG 
MIN_AUDIO_LENGTH = 16000
BUFFER_TRIGGER = 32000   
CHUNK_SIZE = 32000        


#async def stream_tts(ws: WebSocket, text: str):
async def stream_tts(ws: WebSocket, pcm_bytes: bytes):

    try:
        print(f"[TTS] Sending {len(pcm_bytes)} bytes of audio...")

        for i in range(0, len(pcm_bytes), CHUNK_SIZE):
            if ws.client_state != WebSocketState.CONNECTED:
                break
            await ws.send_bytes(pcm_bytes[i:i + CHUNK_SIZE])
            await asyncio.sleep(0.2)

        await ws.send_text("_finish_speech_")
        print("[TTS] Done streaming.")

    except Exception as e:
        print("[TTS STREAM ERROR]", e)

@app.websocket("/chat")
async def websocket_chat(ws: WebSocket):
    await ws.accept()
    print("Client connected")

    conversation_history = ""
    audio_buffer = []

    with mlflow.start_run(run_name =f"session_{id(ws)}") as run:
        chat_agent.log_static_params(chat_agent)
        mlflow.set_tags({"session_id": str(id(ws))})
        # Initial Agent Greeting
        try:
            initial_text = chat_agent.get_initial_message()
            conversation_history += f"Agent: {initial_text}\n"

            #new check
            audio_np, sr = await asyncio.to_thread(tts, initial_text)
            pcm_bytes = tts.float_to_pcm16(audio_np)
            
            await ws.send_text(f"Agent: {initial_text}")
            #asyncio.create_task(stream_tts(ws, initial_text))
            await stream_tts(ws, pcm_bytes)
        except Exception as e:
            print("[INIT TTS ERROR]", e)

        try:
            while True:
                msg = await ws.receive()

                # TEXT message
                if "text" in msg:
                    user_text = msg["text"].strip()
                    if not user_text:
                        continue

                    conversation_history += f"User: {user_text}\n"
                    await ws.send_text(f"User: {user_text}")

                    with mlflow.start_run(nested=True):
                            
                        mlflow.set_tag("request", user_text)
                        mlflow.log_metric("user_input_length", len(user_text))
                        
                        with mlflow.start_span("agent_interaction", span_type=SpanType.AGENT):
                            # Run agent in background
                            response = await chat_agent.generate_response(user_text)
                            print(f"DEBUG: User={user_text}, Response={response}")
                            
                        mlflow.set_tag("response", response)
                        mlflow.log_metric("agent_output_length", len(response))
                            
                            # Log full turn text as artifact
                        turn_log = f"User: {user_text}\nAgent: {response}\n"
                        filename = f"turn_{int(time.time())}.txt"
                        mlflow.log_text(turn_log, filename)

                    conversation_history += f"Agent: {response}\n"
                    #new check
                    audio_np, sr = await asyncio.to_thread(tts, response)
                    pcm_bytes = tts.float_to_pcm16(audio_np)

                    await ws.send_text(f"Agent: {response}")
                    asyncio.create_task(stream_tts(ws, pcm_bytes))

                # AUDIO message
                elif "bytes" in msg:
                    audio_bytes = msg["bytes"]
                    if not audio_bytes:
                        continue

                    audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                    if audio_np.ndim > 1:
                        audio_np = audio_np.mean(axis=1)

                    if len(audio_np) == 0:
                        continue

                    audio_buffer.append(audio_np)
                    total_len = sum(len(a) for a in audio_buffer)

                    # Once buffer has enough audio, process
                    if total_len > BUFFER_TRIGGER:
                        combined_audio = np.concatenate(audio_buffer)
                        audio_buffer = []  # reset buffer
                        print(f"Received {len(combined_audio)} samples for STT...")

                        try:
                            
                            with mlflow.start_span("stt_transcribe", span_type=SpanType.TOOL):
                                user_text = await asyncio.to_thread(stt.transcribe, combined_audio, 16000)
                                
                            if user_text.strip():
                                print(f"Kipla (spoken): {user_text}")
                                conversation_history += f"User: {user_text}\n"
                                await ws.send_text(f"Kipla (spoken): {user_text}")
                                
                                with mlflow.start_run(nested=True):
                                    mlflow.set_tag("request", user_text)
                                    mlflow.log_metric("user_input_length", len(user_text))
                                    
                                    with mlflow.start_span("agent_response", span_type=SpanType.AGENT):

                                        response = await chat_agent.generate_response(user_text)
                                        print(f"response type before len: {type(response)}")

                                    mlflow.set_tag("response", response)
                                    mlflow.log_metric("agent_output_length", len(response))
                                        
                                    turn_log = f"User: {user_text}\nAgent: {response}\n"
                                    filename = f"turn_{int(time.time())}.txt"
                                    mlflow.log_text(turn_log, filename)

                                conversation_history += f"Agent: {response}\n"
                                with mlflow.start_span("tts_synthesis", span_type=SpanType.TOOL):
                                   # Generate TTS new add
                                    audio_np, sr = await asyncio.to_thread(tts, response)
                                
                                pcm_bytes = tts.float_to_pcm16(audio_np)

                                await ws.send_text(f"Agent: {response}")
                                asyncio.create_task(stream_tts(ws, pcm_bytes))

                        except Exception as e:
                            print("[STT ERROR]", e)
                            traceback.print_exc()
                            continue

        except WebSocketDisconnect:
            print("Client disconnected")

        except Exception as e:
            print("[WebSocket ERROR]", e)

        finally:
            if ws.client_state != WebSocketState.DISCONNECTED:
                await ws.close()
            print("WebSocket closed")





# from fastapi import FastAPI, WebSocket, WebSocketDisconnect
# from starlette.websockets import WebSocketState
# import numpy as np
# import io
# import soundfile as sf
# import asyncio
# from src.agent_workflow.agent import ConversationalAgent
# #from src.pipeline.audio.speech_to_text import SpeechToText
# from src.pipeline.audio.transcription import CloudTranscription
# from src.pipeline.audio.cloud_tts import CloudTextToSpeech

# app = FastAPI()

# chat_agent = ConversationalAgent()
# stt = CloudTranscription(
#     endpoint_url="https://api-inference.huggingface.co",  # or your provider endpoint
#     api_key="",
#     model="openai/whisper-large-v3",
#     provider="huggingface",  # adjust if using groq or others
#     chunk=3,
#     target_rate=16000
# )
# #tts = TextToSpeech(provider="hf-inference",api_key="", model="suno/bark-small")

# tts = CloudTextToSpeech(
#     endpoint_url="https://api.groq.com/openai/v1", 
#     api_key="", 
#     model="playai-tts",  # or another available TTS model
#     provider="groq",
#     voice= "Fritz-PlayAI"#"Aaliyah-PlayAI" 
# )


# @app.websocket("/chat")
# async def websocket_chat(ws: WebSocket):
#     await ws.accept()

#     conversation_history = ""
    
#     # initial_text = chat_agent.get_initial_message()
#     # conversation_history = f"Agent: {initial_text}\n"

#     # await ws.send_text(f"Agent: {initial_text}")

#     try:
#         initial_text = chat_agent.get_initial_message()
#         conversation_history = f"Agent: {initial_text}\n"

#         await ws.send_text(f"Agent: {initial_text}")
#         # Stream initial TTS
#         asyncio.create_task(stream_tts(ws, initial_text))


#         while True:
#             msg = await ws.receive()

#             # if msg["type"] != "websocket.receive":
#             #     continue

#             # ----- TEXT message from user -----
#             if "text" in msg:
#                 user_text = msg["text"].strip()
#                 if not user_text:
#                     continue

#                 # Update chat and history
#                 conversation_history += f"User: {user_text}\n"
#                 await ws.send_text(f"User: {user_text}") ##checkk
                

#                 # Agent response
#                 response = await asyncio.to_thread(chat_agent.agent, "conversation_agent", text=conversation_history)
#                 conversation_history += f"Agent: {response}\n"

#                 await ws.send_text(f"Agent: {response}")

#                 # TTS response
#                 # Stream TTS for AI response
#                 if response.strip():
#                     asyncio.create_task(stream_tts(ws, response))

#             elif "bytes" in msg:
#                 audio_bytes = msg["bytes"]
#                 print(f"Received {len(audio_bytes)} bytes of audio from user")  
#                 print(f"Transcribing audio...")

#                 audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
#                 sample_rate = 16000
#                 if audio_np.size == 0:
#                     continue

               
#                 print(f"Audio np shape: {audio_np.shape}, dtype: {audio_np.dtype}")


#                 try:
#                     user_text = await asyncio.to_thread(stt.transcribe, audio_np, sample_rate)#, model="openai/whisper-large-v3"
#                 # except Exception as e:
#                 #     print("[STT ERROR]", e)
#                 #     #await ws.send_text("SYSTEM: Transcription failed.") # Send user feedback
#                 #     continue

#                     if user_text.strip():
                    
#                         print(f"Kipla (spoken): {user_text}")
#                         # Update chat and history
#                         conversation_history += f"User: {user_text}\n"
#                         await ws.send_text(f"Kipla (spoken): {user_text}")
                        
#                         # Agent response
#                         response = await asyncio.to_thread(chat_agent.agent, "conversation_agent", text=conversation_history)
#                         conversation_history += f"Agent: {response}\n"

#                         await ws.send_text(f"Agent: {response}")

#                         if response.strip():

#                             asyncio.create_task(stream_tts(ws, response))
#                 except Exception as e:
#                     print("[STT ERROR]", e)


#     except WebSocketDisconnect:
#         print("Client disconnected")
#     except Exception as e:
#         print("WebSocket error:", e)
 
#     finally:
#         if ws.client_state != WebSocketState.DISCONNECTED:
#             await ws.close()
#         print("WebSocket closed")

# async def stream_tts(ws: WebSocket, text: str):
#     try:
#         audio_np, sr = await asyncio.to_thread(tts, text)
#         pcm_bytes = tts.float_to_pcm16(audio_np)

#         # Break audio into smaller chunks for streaming
#         chunk_size = 32000  # 1-2 seconds
#         for i in range(0, len(pcm_bytes), chunk_size):
#             if ws.client_state != WebSocketState.CONNECTED:
#                 break
#             await ws.send_bytes(pcm_bytes[i:i+chunk_size])
#             await asyncio.sleep(0.05)  # slight delay to allow frontend playback
#     except Exception as e:
#         print("[TTS STREAM ERROR]", e)



