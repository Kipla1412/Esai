"""Onnx Model's Module"""
try:
    import onnxruntime as ort
    
    ONNX_RUNTIME = True
except ImportError:
    ONNX_RUNTIME = False

import numpy as np
import torch
from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig

from transformers.modeling_outputs import SequenceClassifierOutput

from transformers.modeling_utils import PreTrainedModel
from .registry import Registry

class OnnxModel(PreTrainedModel):
    def __init__(self, model, config=None):
        if not ONNX_RUNTIME:
            raise ImportError('onnxruntime is not available - install "model" extra to enable')
        super().__init__(AutoConfig.from_pretrained(config) if config else OnnxConfig())
        self.model = ort.InferenceSession(model, ort.SessionOptions(), self.providers())
        Registry.register(self)

    @property
    def device(self):
        return -1

    def providers(self):
        if torch.cuda.is_available() and "CUDAExecutionProvider" in ort.get_available_providers():
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    def forward(self, **inputs):
        inputs = self.parse(inputs)
        results = self.model.run(None, inputs)
        if any(x.name for x in self.model.get_outputs() if x.name == "logits"):
            return SequenceClassifierOutput(logits=torch.from_numpy(np.array(results[0])))
        return torch.from_numpy(np.array(results))

    def parse(self, inputs):
        features = {}
        for key in ["input_ids", "attention_mask", "token_type_ids"]:
            if key in inputs:
                value = inputs[key]
                if hasattr(value, "cpu"):
                    value = value.cpu().numpy()
                features[key] = np.asarray(value)
        return features

class OnnxConfig(PretrainedConfig):
    pass
