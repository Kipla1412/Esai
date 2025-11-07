

# # from trulens_eval import Tru, TruCustomApp, Feedback
# # from fastapi import FastAPI, WebSocket, WebSocketDisconnect
# # from starlette.websockets import WebSocketState
# # import numpy as np
# # import asyncio
# # from src.agent_workflow.agent_check import ConversationalAgent
# # from src.pipeline.audio.transcription import CloudTranscription
# # from src.pipeline.audio.cloud_tts import CloudTextToSpeech
# # from fastapi.middleware.cors import CORSMiddleware

# # import os
# # os.environ["TRULENS_OTEL_TRACING"] = "1"


# # app = FastAPI()

# # app.add_middleware(
# #     CORSMiddleware,
# #     allow_origins=["*"],
# #     allow_methods=["*"],
# #     allow_headers=["*"],
# # )
# # # initialize agent, STT, TTS
# # chat_agent = ConversationalAgent()
# # stt = CloudTranscription(
# #     endpoint_url="https://api-inference.huggingface.co",
# #     api_key=,
# #     model="openai/whisper-large-v3",
# #     provider="huggingface",
# #     chunk=10,
# #     target_rate=16000
# # )
# # tts = CloudTextToSpeech(
# #     endpoint_url="https://api.groq.com/openai/v1",
# #     api_key=,
# #     model="playai-tts",
# #     provider="groq",
# #     voice="Arista-PlayAI"
# # )


# # # audio config
# # MIN_AUDIO_LENGTH = 16000 # 1 sec @16kHz
# # BUFFER_TRIGGER = 32000    # 2 sec
# # CHUNK_SIZE = 32000        # 32000 bytes for streaming

# # tru = Tru()

# # # Simple feedback metric (example)
# # def voice_agent(conversation_history: str) -> str:
# #     # Run async function in event loop
# #     response = chat_agent.agent("conversation_agent", text=conversation_history)
# #     return {"conversation": conversation_history, "response": response}

# # tru_app = TruCustomApp(
# #     app=voice_agent,
# #     app_id="voice_agent_test"
# # )


# # @app.websocket("/chat")
# # async def websocket_chat(ws: WebSocket):
# #     await ws.accept()
# #     print("Client connected")

# #     conversation_history = ""
# #     audio_buffer = []

  
# #     # Initial Agent Greeting
    
# #     try:
# #         initial_text = chat_agent.get_initial_message()
# #         conversation_history += f"Agent: {initial_text}\n"

# #         #new check
# #         audio_np, sr = await asyncio.to_thread(tts, initial_text)
# #         pcm_bytes = tts.float_to_pcm16(audio_np)
        
# #         await ws.send_text(f"Agent: {initial_text}")
# #         #asyncio.create_task(stream_tts(ws, initial_text))
# #         await stream_tts(ws, pcm_bytes)
# #     except Exception as e:
# #         print("[INIT TTS ERROR]", e)

    
# #     # Main Loop: Receive + Process
   
# #     try:
# #         while True:
# #             msg = await ws.receive()

# #             # TEXT message
# #             if "text" in msg:
# #                 user_text = msg["text"].strip()
# #                 if not user_text:
# #                     continue

# #                 conversation_history += f"User: {user_text}\n"
# #                 await ws.send_text(f"User: {user_text}")

# #                 # Run agent and record with TruLens
# #                 response = await asyncio.to_thread(
# #                     chat_agent.agent, "conversation_agent", text=conversation_history
# #                 )
# #                 conversation_history += f"Agent: {response}\n"
# #                 with tru_app.run(run_name="voice_agent") as rec:
# #                     rec.record_run(
# #                         inputs={"latest_user_text": user_text, "conversation": conversation_history},
# #                         outputs={"response": response}
# #                     )


# #                 await ws.send_text(f"Agent: {response}")

# #                 #new check
# #                 audio_np, sr = await asyncio.to_thread(tts, response)
# #                 pcm_bytes = tts.float_to_pcm16(audio_np)

               
# #                 #asyncio.create_task(stream_tts(ws, response))
# #                 asyncio.create_task(stream_tts(ws, pcm_bytes))

# #             # AUDIO message
# #             elif "bytes" in msg:
# #                 audio_bytes = msg["bytes"]
# #                 if not audio_bytes:
# #                     continue

# #                 audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
# #                 if audio_np.ndim > 1:
# #                     audio_np = audio_np.mean(axis=1)

# #                 if len(audio_np) == 0:
# #                     continue

# #                 audio_buffer.append(audio_np)
# #                 total_len = sum(len(a) for a in audio_buffer)

# #                 # Once buffer has enough audio, process
# #                 if total_len > BUFFER_TRIGGER:
# #                     combined_audio = np.concatenate(audio_buffer)
# #                     audio_buffer = []  # reset buffer
# #                     print(f"Received {len(combined_audio)} samples for STT...")

# #                     try:
# #                         user_text = await asyncio.to_thread(stt.transcribe, combined_audio, 16000)
# #                         if user_text.strip():
# #                             print(f"Kipla (spoken): {user_text}")
# #                             conversation_history += f"User: {user_text}\n"
# #                             await ws.send_text(f"Kipla (spoken): {user_text}")

# #                              # Run agent and record with TruLens
# #                             # 
# #                             # Run agent and record with TruLens
# #                             response = await asyncio.to_thread(
# #                                 chat_agent.agent, "conversation_agent", text=conversation_history
# #                             )
# #                             conversation_history += f"Agent: {response}\n"
# #                             with tru_app.run(run_name="voice_agent") as rec:
# #                                 rec.record_run(
# #                                     inputs={"latest_user_text": user_text, "conversation": conversation_history},
# #                                     outputs={"response": response}
# #                                 )
# #                                # Generate TTS new add
# #                             audio_np, sr = await asyncio.to_thread(tts, response)
# #                             pcm_bytes = tts.float_to_pcm16(audio_np)

# #                             #await ws.send_text(f"Agent: {response}")
# #                                 #asyncio.create_task(stream_tts(ws, response))
# #                             asyncio.create_task(stream_tts(ws, pcm_bytes))

# #                     except Exception as e:
# #                         print("[STT ERROR]", e)
# #                         continue

# #     except WebSocketDisconnect:
# #         print("Client disconnected")

# #     except Exception as e:
# #         print("[WebSocket ERROR]", e)

# #     finally:
# #         if ws.client_state != WebSocketState.DISCONNECTED:
# #             await ws.close()
# #         print("WebSocket closed")


# # # Helper: Stream TTS audio

# # #async def stream_tts(ws: WebSocket, text: str):
# # async def stream_tts(ws: WebSocket, pcm_bytes: bytes):

# #     try:
# #         # audio_np, sr = await asyncio.to_thread(tts, text)
# #         # pcm_bytes = tts.float_to_pcm16(audio_np)

# #         print(f"[TTS] Sending {len(pcm_bytes)} bytes of audio...")

# #         for i in range(0, len(pcm_bytes), CHUNK_SIZE):
# #             if ws.client_state != WebSocketState.CONNECTED:
# #                 break
# #             await ws.send_bytes(pcm_bytes[i:i + CHUNK_SIZE])
# #             await asyncio.sleep(0.2)

# #         await ws.send_text("__finish_speech__")
# #         print("[TTS] Done streaming.")

# #     except Exception as e:
# #         print("[TTS STREAM ERROR]", e)
# import threading
# from fastapi import FastAPI, WebSocket, WebSocketDisconnect
# from starlette.websockets import WebSocketState
# import numpy as np
# import asyncio
# from src.agent_workflow.agent_check import ConversationalAgent
# from src.pipeline.audio.transcription import CloudTranscription
# from src.pipeline.audio.cloud_tts import CloudTextToSpeech
# from fastapi.middleware.cors import CORSMiddleware

# import os
# os.environ["TRULENS_OTEL_TRACING"] = "1"

# # TruLens Eval imports
# from trulens_eval import Tru, TruCustomApp
# from trulens.core import TruSession
# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Initialize agent, STT, TTS
# chat_agent = ConversationalAgent()
# stt = CloudTranscription(
#     endpoint_url="https://api-inference.huggingface.co",
#     api_key="",
#     model="openai/whisper-large-v3",
#     provider="huggingface",
#     chunk=10,
#     target_rate=16000
# )
# tts = CloudTextToSpeech(
#     endpoint_url="https://api.groq.com/openai/v1",
#     api_key=""
#     model="playai-tts",
#     provider="groq",
#     voice="Arista-PlayAI"
# )

# # Audio config
# MIN_AUDIO_LENGTH = 16000  # 1 sec @16kHz
# BUFFER_TRIGGER = 32000     # 2 sec
# CHUNK_SIZE = 32000         # bytes per stream chunk

# # Voice agent wrapper for TruLens
# def voice_agent(conversation_history: str) -> str:
#     response_data = chat_agent.agent("conversation_agent", text=conversation_history)
#     return response_data.get("response", str(response_data))
# # Initialize TruLens
# tru = Tru()
# tru_app = TruCustomApp(voice_agent, app_id="voice_agent_check")

# # Helper: Stream TTS audio
# async def stream_tts(ws: WebSocket, pcm_bytes: bytes):
#     try:
#         for i in range(0, len(pcm_bytes), CHUNK_SIZE):
#             if ws.client_state != WebSocketState.CONNECTED:
#                 break
#             await ws.send_bytes(pcm_bytes[i:i + CHUNK_SIZE])
#             await asyncio.sleep(0.2)
#         await ws.send_text("__finish_speech__")
#     except Exception as e:
#         print("[TTS STREAM ERROR]", e)

# @app.websocket("/chat")
# async def websocket_chat(ws: WebSocket):
#     await ws.accept()
#     print("Client connected")

#     conversation_history = ""
#     audio_buffer = []

#     # Initial Agent Greeting
#     try:
#         initial_text = chat_agent.get_initial_message()
#         conversation_history += f"Agent: {initial_text}\n"

#         audio_np, sr = await asyncio.to_thread(tts, initial_text)
#         pcm_bytes = tts.float_to_pcm16(audio_np)
#         await ws.send_text(f"Agent: {initial_text}")
#         await stream_tts(ws, pcm_bytes)
#     except Exception as e:
#         print("[INIT TTS ERROR]", e)

#     # Main Loop
#     try:
#         while True:
#             msg = await ws.receive()

#             # TEXT message
#             if "text" in msg:
#                 user_text = msg["text"].strip()
#                 if not user_text:
#                     continue

#                 conversation_history += f"User: {user_text}\n"
#                 await ws.send_text(f"User: {user_text}")

#                 # Generate response and record with TruLens
#                 with tru_app.run(run_name="voice_agent") as rec:
#                     response = await chat_agent.generate_response(user_text)
#                     conversation_history += f"Agent: {response}\n"

#                     # Proper TruLens recording
#                     rec.inputs={"latest_user_text": user_text, "conversation": conversation_history},
#                     rec.outputs={"response": response}
                    

#                 await ws.send_text(f"Agent: {response}")

#                 # Generate TTS
#                 audio_np, sr = await asyncio.to_thread(tts, response)
#                 pcm_bytes = tts.float_to_pcm16(audio_np)
#                 asyncio.create_task(stream_tts(ws, pcm_bytes))

#             # AUDIO message
#             elif "bytes" in msg:
#                 audio_bytes = msg["bytes"]
#                 if not audio_bytes:
#                     continue

#                 audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
#                 if audio_np.ndim > 1:
#                     audio_np = audio_np.mean(axis=1)

#                 if len(audio_np) == 0:
#                     continue

#                 audio_buffer.append(audio_np)
#                 total_len = sum(len(a) for a in audio_buffer)

#                 if total_len > BUFFER_TRIGGER:
#                     combined_audio = np.concatenate(audio_buffer)
#                     audio_buffer = []  # reset buffer
#                     print(f"Received {len(combined_audio)} samples for STT...")

#                     try:
#                         user_text = await asyncio.to_thread(stt.transcribe, combined_audio, 16000)
#                         if user_text.strip():
#                             print(f"Kipla (spoken): {user_text}")
#                             conversation_history += f"User: {user_text}\n"
#                             await ws.send_text(f"Kipla (spoken): {user_text}")

#                             with tru_app.run(run_name="voice_agent") as rec:
#                                 response = await chat_agent.generate_response(user_text)
#                                 conversation_history += f"Agent: {response}\n"

#                                 rec.inputs = {"latest_user_text": user_text, "conversation": conversation_history}
#                                 rec.outputs = {"response": response}

#                             await ws.send_text(f"Agent: {response}")

#                             audio_np, sr = await asyncio.to_thread(tts, response)
#                             pcm_bytes = tts.float_to_pcm16(audio_np)
#                             asyncio.create_task(stream_tts(ws, pcm_bytes))

#                     except Exception as e:
#                         print("[STT ERROR]", e)
#                         continue

#     except WebSocketDisconnect:
#         print("Client disconnected")
#     except Exception as e:
#         print("[WebSocket ERROR]", e)
#     finally:
#         if ws.client_state != WebSocketState.DISCONNECTED:
#             await ws.close()
#         print("WebSocket closed")

import requests

headers = {
    "Authorization": "Bearer "
}

response = requests.get("https://huggingface.co/api/models", headers=headers)

print("Status code:", response.status_code)
if response.status_code == 200:
    print("Models accessible:")
    for model in response.json()[:100]:  # print first 10 models
        print(model['modelId'])
else:
    print("Error fetching models:")
    print(response.text)

# import requests
# import base64
# import json

# headers = {
#     "Authorization": "Bearer",
#     "Content-Type": "application/json"
# }

# with open(r"D:\backend\ESAI\src\sample.wav", "rb") as f:
#     audio_bytes = f.read()

# payload = {
#     "model": "openai/whisper-large-v3-turbo",
#     "inputs": base64.b64encode(audio_bytes).decode("utf-8")
# }

# response = requests.post(
#     "https://router.huggingface.co/hf-inference",
#     headers=headers,
#     data=json.dumps(payload)
# )

# print("Status code:", response.status_code)
# if response.status_code == 200:
#     print("Response JSON:", response.json())
# else:
#     print("Error response:")
#     print(response.text)
