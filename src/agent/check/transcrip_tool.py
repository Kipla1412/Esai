

# from smolagents import Tool
# from src.pipeline.audio.transcription import CloudTranscription

# class CloudTranscriptionTool(Tool):
#     """
#     SmolAgents-compatible tool that transcribes audio using various cloud providers
#     (HuggingFace, OpenAI, Groq, etc.)
#     """

#     name = "cloud_transcription"
#     description = "Transcribes audio into text using cloud providers (HF/OpenAI/Groq/etc.)"
#     inputs = {
#         "audio_path": {
#             "type": "string",
#             "description": "Path to the audio file to transcribe"
#         }
#     }
#     output_type = "string"

#     def __init__(self, endpoint_url: str | None = None, api_key: str | None = None,
#                  model: str | None = None, provider: str = "huggingface",
#                  name: str = "cloud_transcription",
#                  description: str = "Transcribes audio into text using cloud providers (HF/OpenAI/Groq/etc.)",
#                  **kwargs):
#         """
#         Accepts endpoint_url as optional and absorbs extra kwargs from agent config.
#         """
#         super().__init__()
#         self.name = name
#         self.description = description

#         if endpoint_url is None:
#             raise ValueError("endpoint_url is required for CloudTranscriptionTool")

#         self.transcriber = CloudTranscription(
#             endpoint_url=endpoint_url,
#             api_key=api_key,
#             model=model,
#             provider=provider
#         )
#     def forward(self, audio_path: str) -> str:
#         """
#         Runs transcription on the given audio file.

#         Args:
#             audio_path (str): Path to the audio file.

#         Returns:
#             str: Transcribed text.
#         """
#         try:
#           result = self.transcriber(audio_path)
#           return result if result else "Warning: Transcription returned empty result."
#         except Exception as e:
#           return f"Error: transcription failed due to {e}"

#     def __call__(self, *args, **kwargs):

#         if args:
#             return self.forward(args[0])
#         elif "audio_path" in kwargs:
#             return self.forward(kwargs["audio_path"])
#         else:
#             raise ValueError("Missing required argument 'audio_path'")

from smolagents import Tool
from src.pipeline.audio.transcription import CloudTranscription

class CloudTranscriptionTool(Tool):
    name = "cloud_transcription"
    description = "Transcribes audio using cloud providers (HF/OpenAI/Groq/etc.)"
    inputs = {"audio_path": {"type": "string", "description": "Path to the audio file"}}
    output_type = "string"

    def __init__(self, config=None, **kwargs):
        super().__init__()
        # Merge config and kwargs
        config = config or {}
        config.update(kwargs)

        self.name = config.get("name", self.name)
        self.description = config.get("description", self.description)

        endpoint_url = config.get("endpoint_url")
        if not endpoint_url:
            raise ValueError("endpoint_url is required for CloudTranscriptionTool")

        self.transcriber = CloudTranscription(
            endpoint_url=endpoint_url,
            api_key=config.get("api_key"),
            model=config.get("model"),
            provider=config.get("provider", "huggingface")
        )

    def forward(self, audio_path: str) -> str:
        try:
            result = self.transcriber(audio_path)
            return result if result else "Warning: empty transcription"
        except Exception as e:
            return f"Error: transcription failed due to {e}"

    def __call__(self, *args, **kwargs):
        if args:
            return self.forward(args[0])
        elif "audio_path" in kwargs:
            return self.forward(kwargs["audio_path"])
        else:
            raise ValueError("Missing required argument 'audio_path'")
