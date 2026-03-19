import torch
import timm
from typing import Any


class ModelWithTemperature(torch.nn.Module):
    """
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """

    def __init__(self, model: torch.nn.Module):
        super().__init__()  # type: ignore
        self.model = model
        self.temperature = torch.nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        logits = self.model(input)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(
            logits.size(0), logits.size(1)
        )
        return logits / temperature


class Model(torch.nn.Module):
    def __init__(self, timm_model_arch: str, num_classes: int) -> None:
        super().__init__()  # type: ignore

        basenet: Any = timm.create_model(  # type: ignore
            model_name=timm_model_arch, num_classes=num_classes
        )
        if not isinstance(basenet, torch.nn.Module):
            raise TypeError(f"Expected a torch.nn.Module, got {type(basenet)} instead.")
        self.net = ModelWithTemperature(basenet)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
