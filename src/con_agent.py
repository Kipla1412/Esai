from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import asyncio
#from src.agent_workflow.agent import ConversationalAgent 
from src.agent_workflow.phoenix_connect import ConversationalAgent # adjust import as neede
from dotenv import load_dotenv
import os
load_dotenv(dotenv_path=r"D:\backend\ESAI\src\.env")


print("OPENROUTER_API_KEY:", os.getenv("OPENROUTER_API_KEY"))
app = FastAPI()
chat_agent = ConversationalAgent()

@app.websocket("/chat")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()

    conversation_history = ""
    await websocket.send_text(chat_agent.get_initial_message())

    try:
        while True:
            user_message = await websocket.receive_text()

            #response = await asyncio.to_thread(chat_agent.agent, "conversation_agent", text=conversation_history + f"User: {user_message}\n")
            response = await chat_agent.generate_response(user_message)

            conversation_history += f"User: {user_message}\nAgent: {response}\n"

            await websocket.send_text(response)

    except WebSocketDisconnect:
        print("Client disconnected")

    except Exception as e:
        print(f"WebSocket error: {e}", flush=True)
    finally:
        await websocket.close()
        print("WebSocket connection closed")