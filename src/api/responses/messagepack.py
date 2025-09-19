from io import BytesIO
from typing import Any

import msgpack
from fastapi import Response
from PIL.Image import Image

class MessagePackResponse(Response):

    media_type ="application/msgpack"
    def render(self, content:Any) ->bytes:

        return msgpack.packp(content, default =MessagePackEncoder())
    
class MessagePackEncoder:

    def __call__(self,o):

        if isinstance(o,Image):

            buffered = BytesIO
            o.save(buffered,format =o.format,quality ="keep")
            o =buffered

        if isinstance(o,BytesIO):
            o =o.getvalue()

        return o