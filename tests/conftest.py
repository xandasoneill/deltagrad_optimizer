import pytest
import torch


@pytest.fixture
def toy_quadratic_target():
    return torch.tensor([3.0, -2.0, 0.5])
