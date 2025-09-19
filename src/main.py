from fastapi import FastAPI
from src.api.routers.agent import router as agent_router

app = FastAPI(title="My Agent API")
app.include_router(agent_router)
