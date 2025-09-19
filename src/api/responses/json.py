import base64
import json 

import fastapi.responses
from io import BytesIO

from typing import Any
from PIL.Image import Image

class JSONEncoder(json.JSONEncoder):
    
    def default(self,o):
        if isinstance(o,Image):

            buffered =BytesIO()
            o.save(buffered,format =o.format,quality ="keep")
            o = buffered

        if isinstance(o,BytesIO):

            o = o.getvalue()

        if isinstance(o,bytes):
            return base64.b64encode(o).decode("utf-8")
        
        return super().default(o)
    
class JSONResponse(fastapi.responses.JSONResponse):

    def render(self,content:Any) ->bytes:

        return json.dumps(content,ensure_ascii =False,
                          allow_nan=False,indent=False,
                          separators=(",",":"),
                          cls=JSONEncoder
                          ).encode("utf-8")
        


