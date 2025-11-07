from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState
import numpy as np
import asyncio
from src.agent_workflow.agent import ConversationalAgent
from src.pipeline.audio.transcription import CloudTranscription
from src.pipeline.audio.cloud_tts import CloudTextToSpeech
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
# initialize agent, STT, TTS
chat_agent = ConversationalAgent()

stt = CloudTranscription(
    endpoint_url="https://api-inference.huggingface.co",
    api_key=os.getenv["HUG_API_KEY"],
    model="openai/whisper-large-v3",
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
MIN_AUDIO_LENGTH = 16000 # 1 sec @16kHz
BUFFER_TRIGGER = 32000    # 2 sec
CHUNK_SIZE = 32000        # 32000 bytes for streaming


#async def stream_tts(ws: WebSocket, text: str):
async def stream_tts(ws: WebSocket, pcm_bytes: bytes):

    try:
        # audio_np, sr = await asyncio.to_thread(tts, text)
        # pcm_bytes = tts.float_to_pcm16(audio_np)

        print(f"[TTS] Sending {len(pcm_bytes)} bytes of audio...")

        for i in range(0, len(pcm_bytes), CHUNK_SIZE):
            if ws.client_state != WebSocketState.CONNECTED:
                break
            await ws.send_bytes(pcm_bytes[i:i + CHUNK_SIZE])
            await asyncio.sleep(0.2)

        await ws.send_text("__finish_speech__")
        print("[TTS] Done streaming.")

    except Exception as e:
        print("[TTS STREAM ERROR]", e)

@app.websocket("/chat")
async def websocket_chat(ws: WebSocket):
    await ws.accept()
    print("Client connected")

    conversation_history = ""
    audio_buffer = []

  
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

    
    # Main Loop: Receive + Process
   
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

                # Run agent in background
                response = await asyncio.to_thread(
                    chat_agent.agent, "conversation_agent", text=conversation_history
                )

                conversation_history += f"Agent: {response}\n"
                #new check
                audio_np, sr = await asyncio.to_thread(tts, response)
                pcm_bytes = tts.float_to_pcm16(audio_np)

                await ws.send_text(f"Agent: {response}")
                #asyncio.create_task(stream_tts(ws, response))
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
                        user_text = await asyncio.to_thread(stt.transcribe, combined_audio, 16000)
                        if user_text.strip():
                            print(f"Kipla (spoken): {user_text}")
                            conversation_history += f"User: {user_text}\n"
                            await ws.send_text(f"Kipla (spoken): {user_text}")

                            response = await asyncio.to_thread(
                                chat_agent.agent, "conversation_agent", text=conversation_history
                            )
                            conversation_history += f"Agent: {response}\n"
                            
                            # Generate TTS new add
                            audio_np, sr = await asyncio.to_thread(tts, response)
                            pcm_bytes = tts.float_to_pcm16(audio_np)

                            await ws.send_text(f"Agent: {response}")
                            #asyncio.create_task(stream_tts(ws, response))
                            asyncio.create_task(stream_tts(ws, pcm_bytes))

                    except Exception as e:
                        print("[STT ERROR]", e)
                        continue

    except WebSocketDisconnect:
        print("Client disconnected")

    except Exception as e:
        print("[WebSocket ERROR]", e)

    finally:
        if ws.client_state != WebSocketState.DISCONNECTED:
            await ws.close()
        print("WebSocket closed")



