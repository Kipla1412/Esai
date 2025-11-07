# text_to_speech_tool.py
# from smolagents import Tool
# from src.pipeline.audio.texttospeech import TextToSpeech  # your TTS pipeline

# class TextToSpeechTool(Tool):
#     """
#     SmolAgents-compatible tool for converting text to speech using HuggingFace TTS.
#     """
#     name = "text_to_speech"
#     description = "Converts text into speech using HuggingFace text-to-speech models."
#     inputs = {
#         "text": {
#             "type": "string",
#             "description": "Text to convert into audio"
#         },
#         "output_file": {
#             "type": "string",
#             "description": "Optional path to save the generated audio file",
#             "nullable": True
#         },
#         "play": {"type": "boolean", "description": "Whether to play audio", "nullable": True},
#     }
#     output_type = "any"  # returns (audio_array, sample_rate)

#     def __init__(self, model, api_key=None, provider=None, sample_rate=24000):
#         """
#         Initialize the TextToSpeechTool with direct arguments.

#         Args:
#             model: HuggingFace TTS model ID.
#             api_key: API key (optional).
#             provider: Provider name (optional, e.g., 'huggingface').
#             sample_rate: Output audio sample rate.
#         """
#         super().__init__()
#         self.model = model
#         self.api_key = api_key
#         self.provider = provider
#         self.sample_rate = sample_rate

#         self.tts = TextToSpeech(
#             model=self.model,
#             api_key=self.api_key,
#             provider=self.provider,
#             sample_rate=self.sample_rate
#         )

#     def forward(self, text: str, output_file: str = None, play: bool = True):
#         """
#         Converts text to speech.

#         Args:
#             text: Input text string.
#             output_file: Optional path to save WAV file.
#             play: If True, plays audio.

#         Returns:
#             Tuple (numpy float32 audio array, sample rate)
#         """
#         try:
#             audio, sr = self.tts(text, output_file=output_file)
#             if play:
#               self.tts.play_audio(audio, sr)
#             return audio, sr
            
#         except Exception as e:
#             return f"Error: Text-to-speech failed due to {e}"

#     def __call__(self, *args, **kwargs):
#         if args:
#             return self.forward(args[0], output_file=kwargs.get("output_file"))
#         elif "text" in kwargs:
#             return self.forward(kwargs["text"], output_file=kwargs.get("output_file"))
#         else:
#             raise ValueError("Missing required argument 'text'")

# text_to_speech_tool.py
from smolagents import Tool
from src.pipeline.audio.texttospeech import TextToSpeech  # your TTS pipeline

class TextToSpeechTool(Tool):
    """
    SmolAgents-compatible tool for converting text to speech using HuggingFace TTS.
    """
    name = "text_to_speech"
    description = "Converts text into speech using HuggingFace text-to-speech models."
    inputs = {
        "text": {
            "type": "string",
            "description": "Text to convert into audio"
        },
        "output_file": {
            "type": "string",
            "description": "Optional path to save the generated audio file",
            "nullable": True
        },
        "play": {"type": "boolean", "description": "Whether to play audio", "nullable": True},
    }
    output_type = "any"  # returns (audio_array, sample_rate)

    def __init__(self, model, api_key=None, provider=None, sample_rate=24000):
        """
        Initialize the TextToSpeechTool with direct arguments.

        Args:
            model: HuggingFace TTS model ID.
            api_key: API key (optional).
            provider: Provider name (optional, e.g., 'huggingface').
            sample_rate: Output audio sample rate.
        """
        super().__init__()
        self.model = model
        self.api_key = api_key
        self.provider = provider
        self.sample_rate = sample_rate

        self.tts = TextToSpeech(
            model=self.model,
            api_key=self.api_key,
            provider=self.provider,
            sample_rate=self.sample_rate
        )

    def forward(self, text: str, output_file: str = None, play: bool = True):
        """
        Converts text to speech. Always returns (audio, sample_rate). 
        On failure, returns (None, 0).
        """
        try:
            audio, sr = self.tts(text, output_file=output_file)
            if play:
                self.tts.play_audio(audio, sr)
            return audio, sr
        except Exception as e:
            print(f"Error: Text-to-speech failed due to {e}")
            return None, 0


    def __call__(self, *args, **kwargs):
        if args:
            return self.forward(args[0], output_file=kwargs.get("output_file"))
        elif "text" in kwargs:
            return self.forward(kwargs["text"], output_file=kwargs.get("output_file"))
        else:
            raise ValueError("Missing required argument 'text'")
        
def text_to_speech(text: str, model: str, api_key: str = None, provider: str = None,
                   output_file: str = None, play: bool = True):
    """
    Simple function wrapper to use TextToSpeechTool without class instantiation.

    Args:
        text (str): Text to convert.
        model (str): HuggingFace TTS model ID.
        api_key (str, optional): API key if needed.
        provider (str, optional): Provider name.
        output_file (str, optional): Path to save audio file.
        play (bool, optional): Play audio after generating.

    Returns:
        Tuple (audio_array, sample_rate) or (None, 0) on error.
    """
    tts_tool = TextToSpeechTool(model=model, api_key=api_key, provider=provider)
    return tts_tool.forward(text=text, output_file=output_file, play=play)

