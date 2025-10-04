import torch
import torch.nn as nn
import pytest
import sys
import os

# Add project root to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from phi_quant import (
    log_phi,
    pow_phi,
    quantize_magnitudes_to_phi,
    cluster_exponents,
    PhiWeightQuantizer,
    PHI,
)
from models import TinyNet
from data import get_mnist_loader

@pytest.fixture
def a_tensor():
    return torch.tensor([0.1, 1.0, PHI, PHI**2, 10.0])

def test_log_phi_and_pow_phi_roundtrip(a_tensor):
    e = log_phi(a_tensor)
    reconstructed = pow_phi(e.round())
    # log_phi(0.1) -> -4.78 -> rounds to -5
    expected = torch.tensor([PHI**-5, PHI**0, PHI**1, PHI**2, PHI**5]).float()
    assert torch.allclose(reconstructed, expected, atol=1e-4)

def test_quantize_magnitudes_to_phi(a_tensor):
    wq_abs, e = quantize_magnitudes_to_phi(a_tensor, ste=False)
    assert torch.allclose(wq_abs, pow_phi(e))
    assert torch.equal(e, torch.tensor([-5., 0., 1., 2., 5.]))

def test_quantize_magnitudes_with_clipping(a_tensor):
    wq_abs, e = quantize_magnitudes_to_phi(a_tensor, e_min=-1, e_max=3, ste=False)
    # e starts as [-5, 0, 1, 2, 5], gets clipped
    assert torch.equal(e, torch.tensor([-1., 0., 1., 2., 3.]))

def test_cluster_exponents():
    e = torch.arange(10).float()
    e_clustered = cluster_exponents(e, cluster_size=3)
    # Pads to 12, groups of 3
    # [0, 1, 2] -> round(1) = 1
    # [3, 4, 5] -> round(4) = 4
    # [6, 7, 8] -> round(7) = 7
    # [9, 0, 0] -> round(3) = 3 -> this is wrong, should be [9, 9, 9] if we pad with the last value.
    # The current implementation pads with zeros, so the last cluster is [9, 0, 0] -> round(3) = 3. Let's test for that.
    # Let's re-calculate: [9, 0, 0] -> mean is 3, rounded is 3. Correct.
    # The original was length 10.
    # Expected: [1,1,1, 4,4,4, 7,7,7, 3]
    # Actually, the padding is only used for calculation and then removed.
    # So the last group is [9], and the padding is [0,0]. The mean is computed on [9,0,0] which is 3.
    # The result is expanded to size 12 then cropped to 10.
    # [1,1,1, 4,4,4, 7,7,7, 3,3,3] -> [1,1,1, 4,4,4, 7,7,7, 3]
    expected = torch.tensor([1., 1., 1., 4., 4., 4., 7., 7., 7., 3.])
    assert torch.equal(e_clustered, expected)

def test_phi_quantizer_per_tensor():
    layer = nn.Linear(10, 1)
    layer.weight.data.fill_(PHI)
    quantizer = PhiWeightQuantizer(layer, per_channel=False)

    quantizer.train() # Enable STE
    Wq, aux = quantizer._quantize_weight_tensor(layer.weight, train_mode=True)
    assert torch.allclose(Wq, torch.full_like(layer.weight, PHI))
    assert torch.all(aux['e'] == 1.0)

def test_phi_quantizer_per_channel():
    layer = nn.Conv2d(2, 1, kernel_size=3)
    layer.weight.data[0, 0, ...].fill_(PHI**2)
    layer.weight.data[0, 1, ...].fill_(PHI**-1)

    quantizer = PhiWeightQuantizer(layer, per_channel=True)
    quantizer.train()

    Wq, aux = quantizer._quantize_weight_tensor(layer.weight, train_mode=True)

    # Check exponents for each channel
    assert torch.all(aux['e'].flatten()[0:9] == 2.0)
    assert torch.all(aux['e'].flatten()[9:18] == -1.0)

    # Check quantized weights
    assert torch.allclose(Wq.flatten()[0:9], torch.full((9,), PHI**2))
    assert torch.allclose(Wq.flatten()[9:18], torch.full((9,), PHI**-1))

def test_export_packed():
    layer = nn.Linear(5, 1, bias=False)
    layer.weight.data = torch.tensor([[0.1, -1.0, PHI, -PHI**2, 0.0]])

    # Test with a defined eps to get predictable results for 0
    quantizer = PhiWeightQuantizer(layer, eps=1e-9)
    packed = quantizer.export_packed()

    # sign is 1 for non-negative
    expected_sign = torch.tensor([[1, 0, 1, 0, 1]], dtype=torch.uint8)
    # log_phi(0.1) -> -5
    # log_phi(1.0) -> 0
    # log_phi(PHI) -> 1
    # log_phi(PHI**2) -> 2
    # log_phi(0.0) w/ eps=1e-9 -> log_phi(1e-9) -> -43
    expected_exp = torch.tensor([[-5, 0, 1, 2, -43]], dtype=torch.int16)

    assert torch.equal(packed['sign_bit'], expected_sign)
    assert torch.equal(packed['exp'], expected_exp)

def test_phi_quantizer_grad_flow():
    layer = nn.Linear(5, 1, bias=False)
    layer.weight.data.uniform_(-1, 1)
    quantizer = PhiWeightQuantizer(layer)

    # Ensure original weight requires grad
    assert layer.weight.requires_grad is True

    # Dummy input and loss
    x = torch.randn(1, 5)
    y = torch.randn(1, 1)

    # Forward pass through quantizer
    quantizer.train()
    out = quantizer(x)
    loss = (out - y).pow(2).sum()

    # Backward pass
    loss.backward()

    # Check if original weight has a gradient
    assert layer.weight.grad is not None
    assert not torch.all(layer.weight.grad == 0)

def test_qat_integration():
    """
    Integration test for a full QAT step.
    """
    net = TinyNet()
    # Wrap a layer for quantization
    net.fc1 = PhiWeightQuantizer(net.fc1, per_channel=False)

    # Get a batch of data
    train_loader = get_mnist_loader(batch_size=4)
    x, y = next(iter(train_loader))

    # Optimizer
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)

    # Get original weight to check for updates
    original_weight = net.fc1.module.weight.clone().detach()

    # Training step
    net.train()
    opt.zero_grad()
    logits = net(x)
    loss = nn.CrossEntropyLoss()(logits, y)
    loss.backward()
    opt.step()

    # Check that the original weight has been updated
    updated_weight = net.fc1.module.weight
    assert not torch.allclose(original_weight, updated_weight)
    assert updated_weight.grad is not None