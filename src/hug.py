from src.pipeline.llm.huggingface import Generator #huggingface-cli login

model_path= 'gpt2'

#hfgen =HFGeneration(model_path)
generator = Generator(path ="gpt2")
prompt ='Once upon a time we want to go '

output = generator(
    text=prompt,
    maxlength=50,
    stream=False,
    stop=None,
    #defaultrole="prompt",
    #stripthink=False
)



print("generated_text :",output)

