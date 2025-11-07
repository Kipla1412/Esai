import torch

from .base import Pipeline

class Tensors(Pipeline):

    def quantize(self, model):

        return torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype =torch.qint8)
    

    def tensor(self, data):

        return torch.tensor(data)
    

    def context(self):

        return torch.no_grad()