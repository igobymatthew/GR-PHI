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

The core quantization logic is in `phi_quant.py`.


⸻

How to use

This repository now includes an interactive script `main.py` that walks you through the quantization process.

To get started, run:
```bash
python main.py
```

The script will guide you to:
1.  **Select layers** of a `TinyNet` model to quantize.
2.  **Configure quantization parameters** for each selected layer, such as `per_channel`, `cluster_size`, and exponent clipping range (`e_min`, `e_max`).
3.  **Choose an action** to perform:
    *   **Train (QAT):** Fine-tune the model with quantization-aware training.
    *   **Export:** Get the packed `sign` and `exponent` tensors for the quantized weights.
    *   **Estimate Entropy:** Calculate the entropy of the exponents to gauge compressibility.

---
*Update: 2025-10-04 08:57:10*
- Restructured project for modularity (`models.py`, `data.py`).
- Added a comprehensive test suite with `pytest` for quantization logic and QAT.
- Enhanced the interactive trainer in `main.py` to use the MNIST dataset.

---
*Update: 2025-10-02 17:17:43*
- Extracted core logic into `phi_quant.py`.
- Created `main.py` for interactive quantization and experimentation.


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
