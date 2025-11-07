import io
import numpy as np
import soundfile as sf
import simpleaudio as sa
from huggingface_hub import InferenceClient
from ..base import Pipeline
from .signal import Signal


class TextToSpeech(Pipeline):
    """
    Cloud-based Text-to-Speech pipeline using HuggingFace InferenceClient,
    with clean separation of configuration and usage.
    """

    def __init__(self, api_key=None, provider=None, model=None, sample_rate=24000):
        """
        Initialize client with API key, provider, and default model.

        Args:
            api_key: API token to authenticate
            provider: optional provider name (e.g. "huggingface", "azure")
            model: default model id for TTS
            sample_rate: desired output sample rate (Hz)
        """
        self.api_key = api_key
        self.provider = provider
        self.model = model
        self.sample_rate = sample_rate

        client_args = {"token": self.api_key}
        if self.provider:
            client_args["provider"] = self.provider

        self.client = InferenceClient(**client_args)

    async def __call__(self, text, model=None, play=False, output_file=None, voice=None):
        """
        Convert text to speech with optional overrides and playback.

        Args:
            text: text string to synthesize
            model: optional model id to override default
            play: bool, if True, play audio immediately
            output_file: path to save audio file (wav)
            voice: optional voice parameter for providers that support it

        Returns:
            Tuple of (numpy float32 audio array, sample rate)
        """
        try:
            m = model or self.model
            if m is None:
                raise ValueError("TTS model must be specified")

            # Call HuggingFace TTS
            out = self.client.text_to_speech(model=m, text=text)

            # Convert response to bytes
            audio_bytes = None
            if isinstance(out, (bytes, bytearray)):
                audio_bytes = bytes(out)
            elif hasattr(out, "read"):
                audio_bytes = out.read()
            elif isinstance(out, dict):
                for key in ["audio", "wav", "speech", "audio_content"]:
                    if key in out:
                        audio_bytes = out[key]
                        break
            if audio_bytes is None:
                raise RuntimeError("Unexpected TTS response")

            # Decode WAV bytes to numpy
            bio = io.BytesIO(audio_bytes)
            data, sr = sf.read(bio, dtype="float32")

            # Resample if needed
            if sr != self.sample_rate:
                data = Signal.resample(data, sr, self.sample_rate)
                sr = self.sample_rate

            # Convert to PCM16 bytes for streaming
            pcm16 = (data * 32767).astype(np.int16).tobytes()
            return pcm16

        except Exception as e:
            print(f"[TTS ERROR] {e}")
            return b""

    

# import io
# import numpy as np
# import soundfile as sf
# import simpleaudio as sa
# from huggingface_hub import InferenceClient
# from ..base import Pipeline
# from .signal import Signal


# class TextToSpeech(Pipeline):
#     """
#     Cloud-based Text-to-Speech pipeline using HuggingFace InferenceClient,
#     with clean separation of configuration and usage.
#     """

#     def __init__(self, api_key=None, provider=None, model=None, sample_rate=24000):
#         """
#         Initialize client with API key, provider, and default model.

#         Args:
#             api_key: API token to authenticate
#             provider: optional provider name (e.g. "huggingface", "azure")
#             model: default model id for TTS
#             sample_rate: desired output sample rate (Hz)
#         """
#         self.api_key = api_key
#         self.provider = provider
#         self.model = model
#         self.sample_rate = sample_rate

#         client_args = {"token": self.api_key}
#         if self.provider:
#             client_args["provider"] = self.provider

#         self.client = InferenceClient(**client_args)

#     def __call__(self, text, model=None, play=False, output_file=None, voice=None):
#         """
#         Convert text to speech with optional overrides and playback.

#         Args:
#             text: text string to synthesize
#             model: optional model id to override default
#             play: bool, if True, play audio immediately
#             output_file: path to save audio file (wav)
#             voice: optional voice parameter for providers that support it

#         Returns:
#             Tuple of (numpy float32 audio array, sample rate)
#         """
#         m = model or self.model
#         if m is None:
#             raise ValueError("TTS model must be specified either at init or call time")

#         params = {"text": text}
#         if voice:
#             params["voice"] = voice

#         # Call HuggingFace text_to_speech endpoint
#         out = self.client.text_to_speech(model=m, text=text)

#         audio_bytes = None
#         if isinstance(out, (bytes, bytearray)):
#             audio_bytes = bytes(out)
#         elif hasattr(out, "read"):
#             audio_bytes = out.read()
#         elif isinstance(out, dict):
#             for key in ["audio", "wav", "speech", "audio_content"]:
#                 if key in out:
#                     audio_bytes = out[key]
#                     break

#         if audio_bytes is None:
#             raise RuntimeError("Unexpected TTS response type; please inspect `out`")
        
#         print(f"[DEBUG] Received output type: {type(out)} length: {len(audio_bytes) if audio_bytes else 0}")

#         try:
#             bio = io.BytesIO(audio_bytes)
#             data, sr = sf.read(bio, dtype="float32")
#         except Exception as e:
#             print(f"[ERROR] Failed to decode audio: {e}")
#             open("debug_audio.bin", "wb").write(audio_bytes)
#             raise RuntimeError("TTS output not WAV format. Saved raw audio to debug_audio.bin")

#         if sr != self.sample_rate:
#             data = Signal.resample(data, sr, self.sample_rate)
#             sr = self.sample_rate

#         if output_file:
#             sf.write(output_file, data, sr)

#         if play:
#             self.play_audio(data, sr)

#         return data, sr

#     def play_audio(self, audio_array, samplerate=None):
#         """
#         Play a numpy float32 audio buffer.

#         Args:
#             audio_array: numpy array with float32 audio data [-1.0,1.0]
#             samplerate: sample rate (Hz)
#         """
#         samplerate = samplerate or self.sample_rate
#         audio_int16 = np.int16(audio_array * 32767)
#         play_obj = sa.play_buffer(audio_int16, 1, 2, samplerate)
#         play_obj.wait_done()

#     def batch_tts(self, texts, prefix="output"):
#         """
#         Batch conversion: save multiple texts as separate WAV files.

#         Args:
#             texts: list of text strings
#             prefix: filename prefix for saved files

#         Returns:
#             list of saved file paths
#         """
#         files = []
#         for i, text in enumerate(texts):
#             filename = f"{prefix}_{i}.wav"
#             self(text, output_file=filename)
#             files.append(filename)
#         return files
