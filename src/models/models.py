

import os
import torch
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from transformers.models.auto.modeling_auto import MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES
from .onnx import OnnxModel

class Models:
    @staticmethod
    def checklength(config, tokenizer):
        if hasattr(config, "config"):
            config = config.config
        if (
            hasattr(config, "max_position_embeddings")
            and tokenizer
            and hasattr(tokenizer, "model_max_length")
            and tokenizer.model_max_length == int(1e30)
        ):
            tokenizer.model_max_length = config.max_position_embeddings

    @staticmethod
    def maxlength(config, tokenizer):
        if hasattr(config, "config"):
            config = config.config
        keys = config.to_diff_dict()
        return config.max_length if "max_length" in keys or not hasattr(tokenizer, "model_max_length") else tokenizer.model_max_length

    @staticmethod
    def deviceid(gpu):
        if isinstance(gpu, torch.device):
            return gpu
        if gpu is None or not Models.hasaccelerator():
            return -1
        if isinstance(gpu, bool):
            return 0 if gpu else -1
        return int(gpu)

    @staticmethod
    def device(deviceid):
        return deviceid if isinstance(deviceid, torch.device) else torch.device(Models.reference(deviceid))

    @staticmethod
    def reference(deviceid):
        return (
            deviceid
            if isinstance(deviceid, str)
            else (
                "cpu"
                if deviceid < 0
                else f"cuda:{deviceid}" if torch.cuda.is_available() else "mps" if Models.hasmpsdevice() else Models.finddevice()
            )
        )

    @staticmethod
    def acceleratorcount():
        return max(torch.cuda.device_count(), int(Models.hasaccelerator()))

    @staticmethod
    def hasaccelerator():
        return torch.cuda.is_available() or Models.hasmpsdevice() or bool(Models.finddevice())

    @staticmethod
    def hasmpsdevice():
        return os.environ.get("PYTORCH_MPS_DISABLE") != "1" and torch.backends.mps.is_available()

    @staticmethod
    def finddevice():
        return next((device for device in ["xpu"] if hasattr(torch, device) and getattr(torch, device).is_available()), None)

    @staticmethod
    def load(path, config=None, task="default", modelargs=None):
        if isinstance(path, bytes) or (isinstance(path, str) and os.path.isfile(path)):
            return OnnxModel(path, config)
        if not isinstance(path, str):
            return path
        models = {
            "default": AutoModel.from_pretrained,
            "question-answering": AutoModelForQuestionAnswering.from_pretrained,
            "summarization": AutoModelForSeq2SeqLM.from_pretrained,
            "text-classification": AutoModelForSequenceClassification.from_pretrained,
            "zero-shot-classification": AutoModelForSequenceClassification.from_pretrained,
        }
        modelargs = modelargs if modelargs else {}
        return models[task](path, **modelargs) if task in models else path

    @staticmethod
    def tokenizer(path, **kwargs):
        return AutoTokenizer.from_pretrained(path, **kwargs) if isinstance(path, str) else path

    @staticmethod
    def task(path, **kwargs):
        config = None
        if isinstance(path, (list, tuple)) and hasattr(path[0], "config"):
            config = path[0].config
        elif isinstance(path, str):
            config = AutoConfig.from_pretrained(path, **kwargs)
        task = None
        if config:
            architecture = config.architectures[0] if config.architectures else None
            if architecture:
                if architecture in MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES.values():
                    task = "vision"
                elif any(x for x in ["LMHead", "CausalLM"] if x in architecture):
                    task = "language-generation"
                elif "QuestionAnswering" in architecture:
                    task = "question-answering"
                elif "ConditionalGeneration" in architecture:
                    task = "sequence-sequence"
        return task
