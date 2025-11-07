# from src.agent.check.transcrip_tool import CloudTranscriptionTool


# tool = CloudTranscriptionTool(
#     endpoint_url="https://api-inference.huggingface.co/models",
#     api_key="",
#     model="openai/whisper-large-v3",
#     provider="huggingface"
# )


# audio_path = r"D:\backend\ESAI\src\sample.wav"  

# # Call the tool
# print("Transcribing...")
# result = tool.forward(audio_path)

# print("\n Result:")
# print(result)

# from src.agent.check.speech_text import SpeechToTextTool

# tool = SpeechToTextTool(
#         model="openai/whisper-large-v3",
#         api_key="",
#         provider= 'fal-ai'
#     )
# print("Transcribing...")
# audio_path = r"D:\backend\ESAI\src\sample.wav"   
# text = tool.forward(audio_path)
# print("Transcribed Text:", text)


# from src.agent.check.text_speech import TextToSpeechTool

# tool = TextToSpeechTool(
#     model="hexgrad/Kokoro-82M",
#     api_key="",
#     provider="fal-ai",
#     sample_rate=24000
# )

# text = "Hello! This is a test of Text-to-Speech."
# audio, sr = tool.forward(text)
# print("Audio generated with sample rate:", sr)

# adjust import path
from src.agent.check.text_speech import text_to_speech
# Example text to convert
sample_text = "Hello, this is a test of the text-to-speech system."

# HuggingFace TTS model ID
model_id = "hexgrad/Kokoro-82M"  # replace with your model
api_key = ""  # if needed, else None
provider = "fal-ai"   # optional

# Optional: save audio file
output_file = "sample_output.wav"

# Call the function
audio, sr = text_to_speech(
    text=sample_text,
    model=model_id,
    play=True
)

# Check results
if audio is not None:
    print(f"Audio generated successfully, sample rate: {sr}")
else:
    print("Failed to generate audio.")
