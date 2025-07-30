
from threading import Thread
from transformers import AutoModelForImageTextToText, TextIteratorStreamer
from ...models import Models
from ..hfpipeline import HFPipeline
from .generation import Generation

class HFGeneration (Generation):

    def __init__(self, path, template = None, **kwargs):

        super().__init__(path, template, **kwargs)
        self.llm = HFLLM(path,**kwargs)

    def isvision(self):
        return isinstance(self.llm.pipeline.model, AutoModelForImageTextToText)
    
    def stream(self, texts, maxlength, stream, stop, **kwargs):
        yield from self.llm(texts, maxlength=maxlength, stream=stream, stop=stop, **kwargs)

class HFLLM(HFPipeline):

    def __init__(self, path=None, quantize=False, gpu=True, model=None, task=None, **kwargs):
        super().__init__(self.task(path, task, **kwargs), path, quantize, gpu, model, **kwargs)

        # intially Tokenizers load 
        self.pipeline.tokenizer = self.pipeline.tokenizer if self.pipeline.tokenizer else Models.tokenizer(path, **kwargs)


    def __call__(self, text, prefix=None, maxlength=512, workers=0, stream=False, stop=None, **kwargs):

        texts = text if isinstance(text, list) else [text]

        if prefix:
            texts = [f"{prefix}{x}" for x in texts]

        args, kwargs = self.parameters(texts, maxlength, workers, stop, **kwargs)
        
        if stream:
            return StreamingResponse(self.pipeline, texts, stop, **kwargs)()
        
        results = [self.extract(result) for result in self.pipeline(*args, **kwargs)]
        return results[0] if isinstance(text, str) else results
    
    def parameters(self, texts, maxlength, workers, stop, **kwargs):
        
        defaults, model = {"max_length": maxlength, "max_new_tokens": None, "num_workers": workers}, self.pipeline.model
       
        if self.pipeline.task == "image-text-to-text":
            defaults["max_length"] = max(maxlength, 2048)
            tokenid = model.generation_config.pad_token_id
            model.generation_config.pad_token_id = tokenid if tokenid else model.generation_config.eos_token_id
            return [], {**{"text": texts, "truncation": True}, **defaults, **kwargs}
        
        if not model.config.pad_token_id:
            tokenid = model.config.eos_token_id
            tokenid = tokenid[0] if isinstance(tokenid, list) else tokenid
            defaults["pad_token_id"] = tokenid
            
            if "batch_size" in kwargs and self.pipeline.tokenizer.pad_token_id is None:
                self.pipeline.tokenizer.pad_token_id = tokenid
                self.pipeline.tokenizer.padding_side = "left"
        
        if stop:
            defaults["tokenizer"] = self.pipeline.tokenizer
        return [texts], {**defaults, **kwargs}
    
    def extract(self, result):

        result = result[0] if isinstance(result, list) else result
        text = result["generated_text"]
        return text[-1]["content"] if isinstance(text, list) else text

    def task(self, path, task, **kwargs):

        mapping = {"language-generation": "text-generation", "sequence-sequence": "text2text-generation", "vision": "image-text-to-text"}
        if path and not task:
            task = Models.task(path, **kwargs)
        return mapping.get(task, "text2text-generation")


class Generator(HFLLM):

    def __init__(self, path=None, quantize=False, gpu=True, model=None, **kwargs):
        super().__init__(path, quantize, gpu, model, "language-generation", **kwargs)

class Sequences(HFLLM):

    def __init__(self, path=None, quantize=False, gpu=True, model=None, **kwargs):
        super().__init__(path, quantize, gpu, model, "sequence-sequence", **kwargs)

class StreamingResponse:

    def __init__(self, pipeline, texts, stop, **kwargs):
        self.stream = TextIteratorStreamer(pipeline.tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=5)
        kwargs["streamer"] = self.stream
        kwargs["stop_strings"] = stop
        self.thread = Thread(target=pipeline, args=[texts], kwargs=kwargs)
        self.length = len(texts)

    def __call__(self):
        
        self.thread.start()
        return self

    def __iter__(self):
        for _ in range(self.length):
            yield from self.stream