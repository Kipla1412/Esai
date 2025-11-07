import numpy as np
import soundfile as sf
import io
from huggingface_hub import InferenceClient
from ..base import Pipeline
from .signal import Signal 


class SpeechToText(Pipeline):
    def __init__(self, provider=None, api_key=None):
        self.provider = provider
        self.api_key = api_key
        #headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        # Since we ALWAYS convert to WAV bytes before sending to STT
        #headers["Content-Type"] = "audio/wav" 

        self.client = InferenceClient(provider=self.provider, token=self.api_key)

    def __call__(self, audio_input, sample_rate=None, model =None):
        """
        Accepts either:
         - raw numpy array + sample_rate explicitly, OR
         - file path / file-like / (audio_array, samplerate) tuple.

        Returns recognized text.
        """
        try:
            # Case 1: raw numpy + sample_rate provided explicitly
            if isinstance(audio_input, np.ndarray) and sample_rate is not None:
                return self._process_raw_audio(audio_input, sample_rate, model)
            
            # Case 2: file path / file-like / tuple, use read()
            else:
                speech = self.read(audio_input)
                return self._process_speech_chunks(speech, model)

        except Exception as e:
            print(f"[STT ERROR] {e}")
            return ""

    def _process_raw_audio(self, audio_np, sample_rate, model):
        # Mono and resample
        if audio_np.ndim > 1:
            audio_np = Signal.mono(audio_np)
        if sample_rate != 16000:
            audio_np = Signal.resample(audio_np, sample_rate, 16000)
            sample_rate = 16000

        # Write to WAV bytes and call inference
        buffer = io.BytesIO()
        sf.write(buffer, audio_np, sample_rate, format="WAV")
        buffer.seek(0)
        wav_bytes = buffer.getvalue()

        result = self.client.automatic_speech_recognition(wav_bytes, model=model)
        text = result.get("text", "").strip()
        return self.clean(text)

    async def _process_speech_chunks(self, speech, model):
        texts = []
        for raw, rate in speech:
            if raw.ndim > 1:
                raw = Signal.mono(raw)
            if rate != 16000:
                raw = Signal.resample(raw, rate, 16000)
                rate = 16000
            
            # Split into 5 second chunks or as needed
            chunk_size = rate * 5
            for start in range(0, len(raw), chunk_size):
                chunk = raw[start: start + chunk_size]
                if len(chunk) == 0:
                    continue
                buffer = io.BytesIO()
                sf.write(buffer, chunk, rate, format="WAV")
                buffer.seek(0)
                wav_bytes = buffer.getvalue()

                result = self.client.automatic_speech_recognition(wav_bytes, model=model)
                text = result.get("text", "").strip()
                if text:
                    texts.append(self.clean(text))

        return " ".join(texts)

    # async def __call__(self, audio_input, model):
    #     """
    #     Accepts raw bytes (PCM or WAV) and returns text.
    #     Designed for small streaming chunks.
    #     """
    #     try:
    #         speech = self.read(audio_input)
    #         texts = []
    #         for raw, rate in speech:
    #             # Optional: process raw audio (mono, resample)
    #             if raw.ndim > 1:
    #                 raw = Signal.mono(raw)

    #             if rate != 16000:
    #                 raw = Signal.resample(raw, rate, 16000)
    #                 rate = 16000

    #         # Chunk audio into 10-second pieces (adjust as needed)
    #             for chunk, chunk_rate in self.segments(raw, rate, chunk=5):
    #                 buffer = io.BytesIO()
    #                 sf.write(buffer, chunk, chunk_rate, format="WAV")
    #                 buffer.seek(0)
    #                 wav_bytes = buffer.getvalue()

    #                 # Call HuggingFace STT per chunk
    #                 result = self.client.automatic_speech_recognition(wav_bytes, model=model)
    #                 text = result.get("text", "").strip()
    #                 if text:
    #                     texts.append(self.clean(text))

    #         return " ".join(texts)

    #     except Exception as e:
    #         print(f"[STT ERROR] {e}")
    #         return ""


    def isaudio(self, audio):
        return isinstance(audio, (str, tuple, np.ndarray)) or hasattr(audio, "read")

    def read(self, audio):
        values = [audio] if self.isaudio(audio) else audio
        speech = []
        for x in values:
            if isinstance(x, str) or hasattr(x, "read"):
                raw, samplerate = sf.read(x)
            elif isinstance(x, tuple):
                raw, samplerate = x
            # elif isinstance(x, np.ndarray):
            # # Assume default sample rate or require explicit sample rate elsewhere
            # # Here just raise error or handle if you can
            #     raise ValueError("Raw numpy array input requires sample rate tuple (audio, sr)")
            else:
                raise ValueError("Invalid audio input")
            speech.append((raw, samplerate))
        return speech

    def segments(self, raw, rate, chunk):
        step = int(rate * chunk)
        return [(raw[i:i + step], rate) for i in range(0, len(raw), step)]

    def speech_to_text(self, audio, samplerate, model):
        """
        You can still use this method for batch STT if needed.
        """
        buffer = io.BytesIO()
        sf.write(buffer, audio, samplerate, format="WAV")
        buffer.seek(0)
        
        # Convert to bytes for HuggingFace InferenceClient
        audio_file = buffer.getvalue()
        result = self.client.automatic_speech_recognition(audio_file, model=model)
        return result.get("text", "")


    def clean(self, text):
        text = text.strip()
        return text.capitalize() if text.isupper() else text


# import numpy as np
# import soundfile as sf
# import io
# from huggingface_hub import InferenceClient
# from ..base import Pipeline
# from .signal import Signal  # your existing signal utility


# class SpeechToText(Pipeline):
#     def __init__(self, provider=None, api_key=None):
#         self.provider = provider
#         self.api_key = api_key

#         self.client = InferenceClient(provider=self.provider, token=self.api_key)

#     async def __call__(self, audio_input, model):
#         """
#         Accepts raw bytes (PCM or WAV) and returns text.
#         Designed for small streaming chunks.
#         """
#         speech = self.read(audio_input)
#         texts = []
#         for raw, rate in speech:
#             # Optional: process raw audio (mono, resample)
#             if raw.ndim > 1:
#                 raw = Signal.mono(raw)
#             if rate != 16000:
#                 raw = Signal.resample(raw, rate, 16000)
#                 rate = 16000

#             # Chunk audio into 10-second pieces (adjust as needed)
#             for chunk, chunk_rate in self.segments(raw, rate, chunk=3):
#                 text = self.speech_to_text(chunk, chunk_rate, model)
#                 texts.append(self.clean(text))
#         return " ".join(texts)

#     def isaudio(self, audio):
#         return isinstance(audio, (str, tuple, np.ndarray)) or hasattr(audio, "read")

#     def read(self, audio):
#         values = [audio] if self.isaudio(audio) else audio
#         speech = []
#         for x in values:
#             if isinstance(x, str) or hasattr(x, "read"):
#                 raw, samplerate = sf.read(x)
#             elif isinstance(x, tuple):
#                 raw, samplerate = x
#             else:
#                 raise ValueError("Invalid audio input")
#             speech.append((raw, samplerate))
#         return speech

#     def segments(self, raw, rate, chunk):
#         step = int(rate * chunk)
#         return [(raw[i:i + step], rate) for i in range(0, len(raw), step)]

#     def speech_to_text(self, audio, samplerate, model):
#         buffer = io.BytesIO()
#         sf.write(buffer, audio, samplerate, format="WAV")
#         buffer.seek(0)
        
#         # Convert to bytes for HuggingFace InferenceClient
#         audio_file = buffer.getvalue()
#         result = self.client.automatic_speech_recognition(audio_file, model=model)
#         return result.get("text", "")


#     def clean(self, text):
#         text = text.strip()
#         return text.capitalize() if text.isupper() else text
