# GR-PHI
toy PyTorch implementation of golden-ratio (φ) logarithmic weight quantization with a straight-through estimator (STE) so you can fine-tune through it. It supports:
	•	Per-tensor or per-channel (out-channel) quantization
	•	Optional exponent clipping [e_{\min}, e_{\max}]
	•	Exponent sharing (“cluster average”) to collapse small groups to a single φ^k
	•	Simple entropy estimate so you can gauge compressibility

You can either (A) wrap layers with a quantized proxy for training, or (B) post-train quantize and export \{\text{sign}, k\}.

⸻

Core idea

For a weight w:
	•	Separate sign and magnitude.
	•	e = \mathrm{round}\big(\log_\phi(|w| + \varepsilon)\big)
	•	\hat w = \mathrm{sign}(w)\cdot \phi^{\mathrm{clip}(e)}
	•	Backprop: STE passes gradients as if the identity (with optional gradient scaling).

⸻

Code

# phi_quant.py
import math
from typing import Tuple, Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

PHI = (1 + 5**0.5) / 2   # golden ratio ~1.61803398875
LOG_PHI = math.log(PHI)

def log_phi(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return torch.log(x.clamp_min(eps)) / LOG_PHI

def pow_phi(e: torch.Tensor) -> torch.Tensor:
    return torch.exp(e * LOG_PHI)

class STEQuantize(torch.autograd.Function):
    """
    Straight-Through Estimator for rounding exponents to integers.
    Forward: round
    Backward: pass-through gradient (identity)
    """
    @staticmethod
    def forward(ctx, e: torch.Tensor):
        return e.round()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # identity straight-through
        return grad_output

def quantize_magnitudes_to_phi(
    w_abs: torch.Tensor,
    e_min: Optional[int] = None,
    e_max: Optional[int] = None,
    eps: float = 1e-12,
    ste: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      wq_abs: quantized magnitude (>=0)
      e: integer exponents used
    """
    e = log_phi(w_abs, eps=eps)
    e = STEQuantize.apply(e) if ste else e.round()

    if e_min is not None or e_max is not None:
        e = e.clamp(e_min if e_min is not None else e.min().item(),
                    e_max if e_max is not None else e.max().item())

    wq_abs = pow_phi(e)  # phi ** e
    return wq_abs, e

def cluster_exponents(
    e: torch.Tensor,
    cluster_size: int = 0,
    dim: int = -1
) -> torch.Tensor:
    """
    Optional "cluster averaging" on exponent grid:
    groups consecutive elements (along dim) into blocks of size cluster_size,
    replaces with the block-average exponent (rounded).
    cluster_size=0 or 1 -> no change.
    """
    if cluster_size in (0, 1):
        return e

    # reshape to blocks along dim
    n = e.shape[dim]
    pad = (cluster_size - (n % cluster_size)) % cluster_size
    if pad > 0:
        pad_shape = list(e.shape)
        pad_shape[dim] = pad
        e = torch.cat([e, e.new_zeros(pad_shape)], dim=dim)

    # group and average
    new_shape = list(e.shape)
    new_shape[dim] = e.shape[dim] // cluster_size
    e_grouped = e.reshape(*new_shape, cluster_size)
    e_avg = e_grouped.mean(dim=-1).round()

    # expand back to original length
    e_expanded = e_avg.unsqueeze(-1).expand_as(e_grouped).reshape(e.shape)
    if pad > 0:
        # remove padding
        index = [slice(None)] * e.dim()
        index[dim] = slice(0, n)
        e_expanded = e_expanded[tuple(index)]
    return e_expanded

class PhiWeightQuantizer(nn.Module):
    """
    Module that quantizes weights of a wrapped layer onto a φ-ladder.
    Use in train mode with STE for QAT; set .eval() for fixed rounding.
    """
    def __init__(
        self,
        module: nn.Module,
        per_channel: bool = True,
        cluster_size: int = 0,
        e_min: Optional[int] = None,
        e_max: Optional[int] = None,
        eps: float = 1e-12,
    ):
        super().__init__()
        self.module = module
        self.per_channel = per_channel
        self.cluster_size = cluster_size
        self.e_min = e_min
        self.e_max = e_max
        self.eps = eps

        # detect a weight tensor to know channel dim (assume Conv/Linear)
        w = self._weight()
        assert w is not None, "Wrapped module must have .weight"
        # For Linear/Conv, out_features/out_channels is dim 0
        self.channel_dim = 0

    def _weight(self) -> Optional[torch.Tensor]:
        return getattr(self.module, "weight", None)

    def _quantize_weight_tensor(self, W: torch.Tensor, train_mode: bool) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # Separate sign and magnitude
        sign = torch.sign(W)
        w_abs = W.abs()

        # Per-channel option: quantize each out-channel independently
        if self.per_channel and W.dim() >= 2:
            # flatten per out-channel
            shape = W.shape
            C = shape[self.channel_dim]
            Wv = W.view(C, -1)
            signv = sign.view(C, -1)
            abs_v = w_abs.view(C, -1)

            e_all = []
            wq_abs_all = []
            for c in range(C):
                wq_abs_c, e_c = quantize_magnitudes_to_phi(
                    abs_v[c],
                    e_min=self.e_min,
                    e_max=self.e_max,
                    eps=self.eps,
                    ste=train_mode,  # STE during train
                )
                # optional exponent clustering inside each channel
                if self.cluster_size and self.cluster_size > 1:
                    e_c = cluster_exponents(e_c, cluster_size=self.cluster_size, dim=-1)
                    wq_abs_c = pow_phi(e_c)

                e_all.append(e_c)
                wq_abs_all.append(wq_abs_c)

            e = torch.stack(e_all, dim=0).view(*shape)
            wq_abs = torch.stack(wq_abs_all, dim=0).view(*shape)
            sign = signv.view(*shape)
        else:
            wq_abs, e = quantize_magnitudes_to_phi(
                w_abs,
                e_min=self.e_min,
                e_max=self.e_max,
                eps=self.eps,
                ste=train_mode,
            )
            if self.cluster_size and self.cluster_size > 1:
                # cluster along last dim as a simple default
                e = cluster_exponents(e, cluster_size=self.cluster_size, dim=-1)
                wq_abs = pow_phi(e)

        Wq = sign * wq_abs
        aux = {"e": e, "sign": sign}
        return Wq, aux

    def forward(self, *args, **kwargs):
        train_mode = self.training
        W = self._weight()
        if W is None:
            return self.module(*args, **kwargs)

        # Quantize a shadow weight and use it for the forward
        Wq, aux = self._quantize_weight_tensor(W, train_mode=train_mode)

        # Use a reparam trick: add and subtract so grads flow to W
        # y = f(Wq + (W - W).detach()) == f(Wq) in forward,
        # but backward dL/dW = dL/dWq (STE already applied on exponent)
        original = self.module.weight
        self.module.weight = nn.Parameter(Wq + (W - W).detach(), requires_grad=True)
        try:
            out = self.module(*args, **kwargs)
        finally:
            # restore original parameter
            self.module.weight = original
        return out

    @torch.no_grad()
    def export_packed(self) -> Dict[str, torch.Tensor]:
        """
        Export sign bit and exponent integers you can entropy-code.
        """
        W = self._weight()
        if W is None:
            return {}

        sign = (W >= 0).to(torch.uint8)  # 1 for non-negative
        e = log_phi(W.abs(), eps=self.eps).round().to(torch.int16)

        if self.e_min is not None or self.e_max is not None:
            e = e.clamp(self.e_min if self.e_min is not None else e.min().item(),
                        self.e_max if self.e_max is not None else e.max().item())

        return {"sign_bit": sign, "exp": e}

    @torch.no_grad()
    def entropy_bits_per_exp(self, bins: Optional[int] = None) -> float:
        """
        Crude histogram entropy of exponents (per symbol, in bits).
        """
        W = self._weight()
        e = log_phi(W.abs(), eps=self.eps).round()
        if self.e_min is not None or self.e_max is not None:
            e = e.clamp(self.e_min if self.e_min is not None else e.min().item(),
                        self.e_max if self.e_max is not None else e.max().item())

        vals, counts = torch.unique(e, return_counts=True)
        p = counts.float() / counts.sum()
        H = -(p * (p.clamp_min(1e-12).log2())).sum().item()
        return float(H)


⸻

How to use

import torch
import torch.nn as nn
from phi_quant import PhiWeightQuantizer

class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 10, bias=False)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

net = TinyNet()

# Wrap layers you want quantized
net.fc1 = PhiWeightQuantizer(
    net.fc1,
    per_channel=True,     # out-channel wise exponents
    cluster_size=32,      # share exponents across blocks of 32 weights
    e_min=-16, e_max=8,   # clip exponent range
)
net.fc2 = PhiWeightQuantizer(net.fc2, per_channel=True, e_min=-12, e_max=8)

# Train (QAT): STE is active in .train()
net.train()
opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=1e-3)

for step in range(100):
    x = torch.randn(64, 784)
    y = torch.randint(0, 10, (64,))
    logits = net(x)
    loss = nn.CrossEntropyLoss()(logits, y)
    opt.zero_grad()
    loss.backward()
    opt.step()

# Eval: fixed rounding (still deterministic)
net.eval()
with torch.no_grad():
    print("Estimated exponent entropy (fc1):", net.fc1.entropy_bits_per_exp())
    packed_fc1 = net.fc1.export_packed()
    # packed_fc1["sign_bit"] and packed_fc1["exp"] are ready for entropy coding.


⸻

Tips & knobs
	•	Per-channel vs per-tensor: per-channel tends to retain accuracy better; per-tensor compresses metadata more.
	•	Exponent range: tighten e_min/e_max after inspecting histograms; this trims extreme values and improves code length.
	•	Cluster size: larger clusters → more sharing → better compression, higher distortion. Start with 16–64.
	•	Calibration / QAT: do a short QAT fine-tune (like above) after wrapping to recover accuracy.
	•	Export: store {sign_bit, exp} and entropy-code exp layer-wise. Reconstruct with sign * phi**exp.

⸻

What to compare against
	•	Per-channel int8 and int4 (PTQ and QAT)
	•	K-means weight sharing (codebook of 16–256 centers)
	•	Log-base-2 quantization (swap φ→2) to see if φ really helps your distribution
	•	Low-rank or LoRA+quant combos if you’re doing bigger models

If you want, I can add a tiny ANS/Huffman reference for the exp stream and a Conv2d example with per-channel exponent stats.
