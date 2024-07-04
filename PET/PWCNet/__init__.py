import torch
from .run import Network, Extractor, Decoder, Refiner


def build_pwc(model_path):
    return torch.load(model_path)