# speech_to_text_tool_ready.py
from src.pipeline.audio.speech_to_text import SpeechToText
from smolagents import Tool
import os

class SpeechToTextTool(Tool):
    """
    Standalone SmolAgents-style tool for audio → text using HuggingFace ASR.
    """
    name = "speech_to_text"
    description = "Transcribes audio into text using HuggingFace automatic speech recognition."
    inputs = {
        "audio_path": {"type": "string", "description": "Path to the audio file to transcribe"},
        "model": {
            "type": "string",
            "description": "Speech-to-text model identifier.",
            "nullable": True}
    }

    output_type = "string"

    def __init__(self, model: str, api_key: str, provider: str = "huggingface"):
        self.model = model
        self.stt = SpeechToText(provider=provider, api_key=api_key)

    def forward(self, audio_path: str) -> str:
        """
        SmolAgents expects a `forward` method for execution.
        """
        if not os.path.exists(audio_path):
            return f"Error: audio file not found: {audio_path}"
        try:
            return self.stt(audio_path, model=self.model)
        except Exception as e:
            return f"Error: transcription failed due to {e}"

    def __call__(self, *args, **kwargs) -> str:
        """
        Allow calling like a normal function.
        """
        if args:
            return self.forward(args[0])
        elif "audio_path" in kwargs:
            return self.forward(kwargs["audio_path"])
        else:
            return "Error: missing required argument 'audio_path'"

def speech(audio_path: str, model: str = "openai/whisper-large-v3", api_key: str = "") -> str:
    """
    Convert a given audio file to text using ASR models (like Whisper).

    Args:
        audio_path (str): Path to the audio file to transcribe.
        model (str): Model name for transcription.
        api_key (str): API key for the provider.

    Returns:
        str: Transcribed text.
    """
    tool = SpeechToTextTool(model=model, provider="fal-ai", api_key= api_key)
    return tool.forward(audio_path)

# # src/agent/tools/speech_to_text_tool.py
# from smolagents import Tool
# from src.pipeline.audio.speech_to_text import SpeechToText

# class SpeechToTextTool(Tool):
#     """
#     Converts speech audio input into text using HuggingFace InferenceClient.
#     """
#     name = "speech_to_text"
#     description = "Converts speech audio input into text."
#     inputs = {
#         "audio_input": {
#             "type": "string",
#             "description": "Path or file-like object for the audio input."
#         },
#         "model": {
#             "type": "string",
#             "description": "Speech-to-text model identifier.",
#             "nullable": True
#         }
#     }
#     output_type = any

#     def __init__(self, **config):
#         super().__init__()
#         self.provider = config.get("provider")
#         self.api_key = config.get("api_key")
#         self.model = config.get("model")
#         self.stt = SpeechToText(provider=self.provider, api_key=self.api_key)

#     def forward(self, audio_input: str, model: str = None):
#         """
#         Converts the given audio input to text.
#         """
#         model = model or self.model
#         return self.stt(audio_input, model)