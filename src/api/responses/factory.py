from .json import JSONResponse
from .messagepack import MessagePackResponse

class ResponseFactory:

    @staticmethod
    def create(request):

        accept =request.headers.get("accept")
        return MessagePackResponse if accept == MessagePackResponse.media_type else JSONResponse
