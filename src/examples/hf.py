from src.pipeline.hfpipeline import HFPipeline

path = "gpt2"  # HF Hub ID
task = "text-generation"

pipe = HFPipeline(task=task, path=path)
print(pipe.pipeline("Today is a great day because"))