import torch
from src.models.model import MyAwesomeModel

test_tensor = torch.rand(1,1,28,28)
model = MyAwesomeModel()

def test_input_shape():
    assert test_tensor.shape == (1, 1, 28, 28), "Incorrect shape for input tensor"

def test_model_output_shape():
    assert model(test_tensor).shape == (1,10), "Incorrect shape for model output"


