from typing import Optional
from fastapi import APIRouter,Body
from fastapi.responses import StreamingResponse
from .. import application
from ..route import EncodingAPIRoute

router =APIRouter(route_class =EncodingAPIRoute)

@router.post("/agent")
def agent(name: str =Body(...), text: str =Body(...), maxlength: Optional[int] = Body(default =None), stream: Optional[bool] =Body(default =None)):

    kwargs ={key: value for key, value in [("stream",stream),("maxlength", maxlength)] if value}
    result =application.get().agent(name,text,**kwargs)

    return StreamingResponse(result) if stream else result

