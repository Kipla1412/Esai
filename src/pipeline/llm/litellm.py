from transformers.utils import cached_file
import sounddevice as sd
import numpy as np
import soundfile as sf
import io

try :

    import litellm as api

    LITELLM =True
except ImportError:

   LITELLM = False
from .generation import Generation

class LiteLLM(Generation):

    @staticmethod
    def ismodel(path):

        if isinstance(path,str) and LITELLM:

            debug = api.suppress_debug_info

            try:

                api.suppress_debug_info = True
                return api.get_provider_llm(path) and not LiteLLM.ishub(path)
            except:
                return False
            finally:

                api.suppress_debug_info = debug
        return False
    
    @staticmethod

    def ishub(path):

        try:
            return cached_file(path_or_repo_id=path, file_name = "config.json") is not None if "/" in path else False
        except:

            return False
        
    def __init__(self, path,template = None,**kwargs):
        super().__init__(path,template,**kwargs)

        if not LITELLM:
            raise ImportError('LiteLLM is not available - install "pipeline" extra to enable')
        
        self.kwargs = {k:v for k,v in self.kwargs.items() if k not in ["quantize", "gpu", "model", "task"]}

    def text_to_speech(self, text, voice=None , **kwargs):

        """
        Generate speech audio from text and return raw audio bytes.

        Args:
           text(str): Input text for TTs.
           voice (str): Voice model name.
           kwargs: Additional TTs parameters.

        Returns:
           bytes: Raw audio bytes of the synthesized speech.
        
        """
        response = api.speech(
            model =self.path,
            voice = voice,
            input =text,
            **kwargs

        )
        return response.read()
    
    def play_audio(self, audio_bytes):
        
        """Play audio bytes (auto-detect format)"""
        audio, sr = sf.read(io.BytesIO(audio_bytes))
        sd.play(audio, samplerate=sr)
        sd.wait()
    
    def speech_to_text(self, audio_input, **kwargs):
        """
        Transcribe speech audio input to text using LiteLLM.

        Args: 
           audio_input (str or bytes): path to audio file or raw audio bytes.
           kwargs: Additional transcription parameters

        Returns:
           str: Transcribed text.

        """
        result = api.transcription(
            model =self.path,
            file =audio_input,
            **kwargs
        )

        return result.get("text") or result.get("output", "")
     
    def record_audio(self,duration=5, fs=16000):

        print(f"Recording audio for {duration} seconds...")
        audio = sd.rec(int(duration * fs), samplerate= fs, channels=1, dtype='int16')
        sd.wait()
        print("Recording complete.")
        buf = io.BytesIO()
        sf.write(buf, audio, fs, format="WAV")
        buf.seek(0)
        return buf.read() 
        

    def stream(self,texts,maxlength,stream,stop,**kwargs):

        for text in texts:

            result = api.completion(
                model = self.path,
                messages = [{"content": text, "role": "prompt"}] if isinstance(text,str) else text,
                max_tokens =maxlength,
                stream = stream,
                stop =stop,
                **{**self.kwargs,**kwargs}

            )

            yield from self.response(result if stream else [result])
    