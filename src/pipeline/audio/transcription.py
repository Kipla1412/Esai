import numpy as np
import soundfile as sf
import requests
import io
import json
import base64
from .signal import Signal

class CloudTranscription:

    def __init__(self, endpoint_url =None, api_key =None, model =None, provider="huggingface", chunk =10, target_rate = 16000, **kwargs):

        """
        Args:
            endpoint_url (str): Cloud API transcription URL
            api_key (str): Bearer token for authorization (optional)
            model (str): Model name or ID if API supports model selection (optional)
            chunk (int): Chunk size in seconds to split audio for transcription
            target_rate (int): Target sample rate for cloud API compatibility
           
        """
        if endpoint_url is None:
            
            raise ValueError("endpoint_url is required")
        
        self.endpoint_url = endpoint_url
        self.api_key = api_key
        self.model = model
        self.chunk = chunk
        self.provider = provider.lower()
        self.target_rate = target_rate
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    def __call__(self, audio_input, rate=None, join=True):

        return self.transcribe(audio_input, rate=rate, join=join)

    def isaudio(self,audio):

        """
        Checks if input is a single audio element.

        Accepts numpy arrays, file paths, or file-like objects.
        """

        return isinstance(audio,(str, tuple,np.ndarray)) or hasattr(audio, "read")
    
    def read(self, audio, rate):

        """
        Reads audio files or arrays and returns lisk of (audio, sample_rate)  tuples.
        Supports file paths, file-like objects, tuples (array, rate), or raw arrays with given rate.
        """

        speech = []

        values =[audio] if self.isaudio(audio) else audio
        for x in values:

            if isinstance(x, str) or hasattr(x, "read"):
                raw, samplerate = sf.read(x)

            elif isinstance(x, tuple):
                raw, samplerate = x

            else:
                if rate is None:
                    raise ValueError("Must provide sample rate for raw numpy arrays")
                
                raw, samplerate =x, rate
            
            speech.append((raw,samplerate))

        return speech
    
    def segments(self, raw, rate):

        """
        Split raw audio into chuck-sized segments.
        """
        step = int(rate * self.chunk)
        return [(raw[i : i + step], rate)for i in range(0, len(raw), step)]
    
    def transcribe_chunk(self, raw, rate):

        mono = Signal.mono(raw) if raw.ndim > 1 else raw
        resampled = Signal.resample(mono, rate, self.target_rate)

        buffer = io.BytesIO()
        sf.write(buffer, resampled, self.target_rate, format="WAV")
        buffer.seek(0)
        audio_bytes = buffer.read()

        if self.provider == "groq":
            buffer.seek(0)
            files = {"file": ("audio.wav", buffer, "audio/wav")}
            data = {"model": self.model} if self.model else {}
            url = f"{self.endpoint_url}/audio/transcriptions"
            response = requests.post(url, headers=self.headers, data=data, files=files)

        elif self.provider == "huggingface":

            url = f"https://api-inference.huggingface.co/models/{self.model}"
            #url = self.endpoint_url
            headers = {**self.headers, "Content-Type": "audio/wav"}
            response = requests.post(url, headers=headers, data=audio_bytes)
            #response = requests.post(url, headers=headers, data=json.dumps(payload))

        elif self.provider in ["openai", "openrouter", "together", "gemini"]:

            url = f"{self.endpoint_url}"  # usually endpoint + model path
            payload = {
                "model": self.model,
                "audio": base64.b64encode(audio_bytes).decode("utf-8")
            }
            
            response = requests.post(
                url,
                headers={**self.headers, "Content-Type": "application/json"},
                data=json.dumps(payload)
            )

        else:
            raise ValueError(f"Unknown provider {self.provider}")

        if response.status_code != 200:
            raise RuntimeError(
                f"Cloud transcription failed ({self.provider}): {response.status_code} {response.text}"
            )

        result = response.json()
        text = result.get("text") or result.get("output") or ""
        return text.strip() if text else "Warning: empty transcription"

    def batchprocess(self, raw, rate):

        """ Chunk audio, transcribe each separately, return list of dicts with text and raw audio"""

        results =[]

        for segment, seg_rate in self.segments(raw, rate):

            text = self.transcribe_chunk(segment, seg_rate)
            cleaned = self.clean(text)
            results.append({"text": cleaned, "raw": segment, "rate": rate})

        return results
    
    def transcribe(self, audio, rate=None, join=True):
        """
        Main method: accepts flexible audio input types.
        If join=True, returns concatenated text; else detailed chunk-wise list.
        """
        speech = self.read(audio, rate)

        if join:

            texts =[]
            for raw, rate in speech:

                for chunk , seg_rate in self.segments(raw, rate):
                    texts.append(self.clean(self.transcribe_chunk(chunk, seg_rate)))

            return " ".join(texts)
        
        else:
            results = []
            for raw, rate in speech:

                results.append(self.batchprocess(raw, rate))

            return results
        
    def clean(self, text):
        """Clean and normalize transcription string."""

        text = text.strip()
        return text.capitalize() if text.isupper() else text
    


    
         




                