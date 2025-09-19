import os
import sys
import inspect

from fastapi import APIRouter,FastAPI
from .base import API
from ..app import Application
from .factory import APIFactory

def get():

    return INSTANCE

def create():

    return FastAPI(lifespan=lifespan)

def lifespan(application):
    print("server is started...")

    global INSTANCE

    config =Application.read(os.environ.get("CONFIG"))
    api = os.environ.get("API_CLASS")
    INSTANCE =APIFactory.create(config,api) if api else API(config)
    print("Instance created:", INSTANCE)
    print("Application ID:", id(application))

    routers =apirouters()

    print("available routers:",routers)
    for name,router in routers.items():
        if name in config:
            application.include_router(router)

    yield
    print("application end")

def apirouters():

    api =sys.modules[".".join(__name__.split(".")[:-1])]
    available ={}

    for name,rclass in inspect.getmembers(api,inspect.ismodule):
        if hasattr(rclass, "router") and isinstance(rclass.router, APIRouter):

            available[name.lower()] =rclass.router

    return available
    print("Available routers:", available)


def start():

    list(lifespan(app))

app=create()
INSTANCE = None
