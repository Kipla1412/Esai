from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState
import numpy as np
import asyncio
from src.dspy_agent.dspy_agent import ConversationalAgent
from src.pipeline.audio.transcription import CloudTranscription
from src.pipeline.audio.cloud_tts import CloudTextToSpeech
from fastapi.middleware.cors import CORSMiddleware
import mlflow
from mlflow.entities import SpanType
import time
import os
import traceback
import logging
from dotenv import load_dotenv
mlflow.set_experiment("voice_agent_using_dspy")
mlflow.dspy.autolog()

logger = logging.getLogger("VoiceAgent")
logger.setLevel(logging.INFO)
load_dotenv()
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

MIN_AUDIO_LENGTH = 16000
BUFFER_TRIGGER = 32000   
CHUNK_SIZE = 32000        

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

    if mlflow.active_run() is not None:
        mlflow.end_run()

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
            await stream_tts(ws, pcm_bytes)
        except Exception as e:
            logger.exception("[INIT TTS ERROR]")
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
                            
                            try:
                                response = await chat_agent.generate_response(user_text)
                            except Exception as e:
                                logger.error(f"[Agent Error] {e}")
                                response = "I'm sorry, something went wrong while generating a response."
                                mlflow.set_tag("agent_error", str(e))

                        mlflow.set_tag("response", response)
                        mlflow.log_metric("agent_output_length", len(response))
                            
                            # Log full turn text as artifact
                        turn_log = f"User: {user_text}\nAgent: {response}\n"
                        filename = f"turn_{int(time.time())}.txt"
                        mlflow.log_text(turn_log, filename)

                    conversation_history += f"Agent: {response}\n"
                    try:
                        #new check
                        audio_np, sr = await asyncio.to_thread(tts, response)
                        pcm_bytes = tts.float_to_pcm16(audio_np)

                        await ws.send_text(f"Agent: {response}")
                        asyncio.create_task(stream_tts(ws, pcm_bytes))

                    except Exception as tts_err:
                        logger.exception("[TTS ERROR]")
                        await ws.send_text(f"Agent (text only): {response}")
                        mlflow.set_tag("tts_error", str(tts_err))

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
                        audio_buffer = []  

                        logger.info(f"Received {len(combined_audio)} samples for STT...")
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
                                    try:
                                        audio_np, sr = await asyncio.to_thread(tts, response)
                                        pcm_bytes = tts.float_to_pcm16(audio_np)
                                        await ws.send_text(f"Agent: {response}")
                                        asyncio.create_task(stream_tts(ws, pcm_bytes))

                                    except Exception as tts_err:
                                        logger.exception("[TTS ERROR in audio response]")
                                        await ws.send_text(f"Agent (text only): {response}")
                                        mlflow.set_tag("tts_error", str(tts_err))

                        except Exception as e:
                            logger.exception("[STT ERROR]")
                            traceback.print_exc()
                            mlflow.set_tag("stt_error", str(e))
                            continue

        except WebSocketDisconnect:
            print("Client disconnected")

        except Exception as e:
            print("[WebSocket ERROR]", e)

        finally:
            if ws.client_state != WebSocketState.DISCONNECTED:
                await ws.close()
            print("WebSocket closed")

