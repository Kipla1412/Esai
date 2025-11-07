import requests
import base64
import json
import io
import soundfile as sf
import numpy as np
from .signal import Signal  # Your resample helper

class CloudTextToSpeech:
    """
    Cloud-based Text to Speech Wrapper.
    Supports Hugging Face, OpenAI/OpenRouter, and Groq.
    """

    def __init__(self, endpoint_url, api_key=None, model=None, provider="huggingface", target_rate=22050, voice=None):
        self.endpoint_url = endpoint_url
        self.api_key = api_key
        self.model = model
        self.provider = provider.lower()
        self.target_rate = target_rate
        self.voice = voice or "Aaliyah-PlayAI"
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    def __call__(self, text):
        return self.synthesize(text)

    def synthesize(self, text):
        text = text.strip()
        if not text:
            raise ValueError("Input text must be non-empty")

        audio_bytes = None

        if self.provider == "huggingface":
            url = f"{self.endpoint_url}/{self.model}"
            payload = {"inputs": text}
            response = requests.post(url, headers={**self.headers, "Content-Type": "application/json"}, data=json.dumps(payload))
            if response.status_code != 200:
                raise RuntimeError(f"HuggingFace TTS failed: {response.status_code} {response.text}")
            audio_bytes = response.content

        elif self.provider == "groq":
            url = f"{self.endpoint_url}/audio/speech"
            payload = {"model": self.model, "input": text, "voice": self.voice, "response_format": "wav"}
            response = requests.post(url, headers={**self.headers, "Content-Type": "application/json"}, data=json.dumps(payload))
            if response.status_code != 200:
                raise RuntimeError(f"Groq TTS failed: {response.status_code} {response.text}")
            audio_bytes = response.content

        elif self.provider in ["openai", "openrouter"]:
            url = self.endpoint_url
            payload = {"model": self.model, "input": text}
            response = requests.post(url, headers={**self.headers, "Content-Type": "application/json"}, data=json.dumps(payload))
            if response.status_code != 200:
                raise RuntimeError(f"{self.provider} TTS failed: {response.status_code} {response.text}")
            try:
                result = response.json()
                audio_bytes = base64.b64decode(result.get("audio", result.get("data", "")))
            except:
                audio_bytes = response.content

        if not audio_bytes:
            raise RuntimeError("TTS returned empty audio.")

        return self.bytes_to_array(audio_bytes)

    def bytes_to_array(self, audio_bytes, make_mono =True):

        buffer = io.BytesIO(audio_bytes)
        audio, sr = sf.read(buffer, dtype='float32')
        
        if audio.size == 0:
            raise RuntimeError("Audio read successfully but it's empty")
        
        if sr != self.target_rate:
            audio = Signal.resample(audio, sr, self.target_rate)
            sr = self.target_rate
        ## additionally i add this make mono , audio size method
        if make_mono and audio.ndim > 1:
            audio = Signal.mono(audio)
        return audio, sr
    
    def float_to_pcm16(self, audio: np.ndarray):
        """Convert float32 audio (-1 to 1) to PCM16 bytes for WebSocket."""
        audio_int16 = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
        return audio_int16.tobytes()




# import requests
# import base64
# import json
# import io
# import soundfile as sf
# import sounddevice as sd
# import numpy as np
# # ADDED scipy import for robust WAV reading
# import scipy.io.wavfile as wavfile 
# from .signal import Signal

# # ---------------------------------------------------------------
# class CloudTextToSpeech:
#     """
#     Cloud-based Text to Speech Wrapper
#     Supports Hugging Face, OpenAI/OpenRouter, and Groq.
#     """

#     def __init__(self, endpoint_url, api_key=None, model=None, provider="huggingface", target_rate=22050, voice=None):
#         self.endpoint_url = endpoint_url
#         self.api_key = api_key
#         self.model = model
#         self.provider = provider.lower()
#         self.target_rate = target_rate
#         self.voice = voice or "Aaliyah-PlayAI"
#         self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

#     def __call__(self, text,stream=False):
#         if stream:
#             yield self.synthesize(text)
#         else:
#             return self.synthesize(text)

#     def synthesize(self, text):
#         """
#         Core TTS function.
#         Returns: (audio_numpy_array, sample_rate)
#         """
#         text = text.strip()
#         if not text or not isinstance(text, str):
#             raise ValueError("Input text must be a non-empty string")

#         if self.provider == "huggingface":
#             url = f"{self.endpoint_url}/{self.model}"
#             headers = {**self.headers, "Content-Type": "application/json"}
#             payload = {"inputs": text}
#             response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=30)
#             if response.status_code != 200:
#                 raise RuntimeError(f"TTS failed ({self.provider}): {response.status_code} {response.text}")
#             audio_bytes = response.content
#             return self.bytes_to_array(audio_bytes)
#         elif self.provider == "groq":
#             url = f"{self.endpoint_url}/audio/speech"
#             payload = {
#                 "model": self.model,
#                 "input": text,
#                 "voice": self.voice,
#                  "response_format": "wav"}
#             headers = {**self.headers, "Content-Type": "application/json"}

#             response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=30)
#             print(f"[Groq] Status code: {response.status_code}")

#             if response.status_code != 200:
#                 print(f"[Groq] Error: {response.text}")
#                 try:
#                     error_json = response.json()
#                     raise RuntimeError(f"Groq TTS failed: {response.status_code} {error_json.get('error', {}).get('message', response.text)}")
#                 except json.JSONDecodeError:
#                     raise RuntimeError(f"Groq TTS failed: {response.status_code} {response.text}")

#             audio_bytes = response.content

#             # Save audio bytes to file for inspection
#             with open("output_groq.wav", "wb") as f:
#                 f.write(audio_bytes)
#             print("Groq audio saved to output_groq.wav")

#             if not audio_bytes:
#                 raise ValueError("Groq TTS returned empty audio. Check model, voice config, or credentials.")

#             return self.bytes_to_array(audio_bytes)



#         elif self.provider in ["openai", "openrouter"]:
#             url = self.endpoint_url
#             payload = {"model": self.model, "input": text}
#             response = requests.post(
#                 url,
#                 headers={**self.headers, "Content-Type": "application/json"},
#                 data=json.dumps(payload),
#                 timeout=30
#             )
#             if response.status_code != 200:
#                 raise RuntimeError(f"{self.provider} TTS failed: {response.status_code} {response.text}")

#             try:
#                 result = response.json()
#                 # OpenRouter/OpenAI usually returns base64 in a 'data' or 'audio' key
#                 audio_bytes = base64.b64decode(result.get("audio", result.get("data", "")))
#             except json.JSONDecodeError:
#                 audio_bytes = response.content
            
#         else:
#             raise ValueError(f"Unknown provider: {self.provider}")
            
#         if not audio_bytes:
#             raise RuntimeError(
#                 "TTS returned empty audio. Check your API key, model, and voice. "
#                 f"Provider: {self.provider}"
#             )
#         return self.bytes_to_array(audio_bytes)
    
#     # ---------------------------------------------------------------
#     # CORRECTED bytes_to_array method using scipy.io.wavfile
#     # ---------------------------------------------------------------
#     def bytes_to_array(self, audio_bytes):
#         if not audio_bytes:
#             raise ValueError("No audio bytes to convert")
#         buffer = io.BytesIO(audio_bytes)
#         try:
#             audio, sr = sf.read(buffer, dtype='float32')
#         except RuntimeError as e:
#             raise RuntimeError(f"Failed to read audio bytes with soundfile: {e}")

#         if audio.size == 0:
#             raise RuntimeError("Audio read successfully but it's empty")

#         if sr != self.target_rate:
#             audio = Signal.resample(audio, sr, self.target_rate)
#             sr = self.target_rate

#         return audio, sr

# # ---------------------------------------------------------------
# # Example usage (change provider below)
# if __name__ == "__main__":
    
#     # 🎯 Example 2 — Groq (if you want to switch)
#     tts = CloudTextToSpeech(
#         endpoint_url="https://api.groq.com/openai/v1",
#         api_key="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", # Use your actual key
#         model="playai-tts",
#         provider="groq",
#         voice="Aaliyah-PlayAI" 
#     )

#     text = "Hello Umar, I am testing the new audio decoder using scipy.io.wavfile."
#     print("Generating speech...")
#     try:
#         audio, rate = tts(text)
#         print(f"Received audio array of shape {audio.shape} at rate {rate} Hz.")
#         print("Playing audio...")
#         sd.play(audio, rate)
#         sd.wait()
#         print("✅ Done")
#     except Exception as e:
#         print("Error generating TTS:", e)

