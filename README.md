# Neural Steganography (STEGASURAS & Emu3 Extensions)

Steganography via Arithmetic Coding (AC) and modern Multimodal / Language Models.

This project extends neural linguistic steganography ([Ziegler et al., 2019](https://arxiv.org/abs/1909.01496)) to multimodal generative models. Using autoregressive token generation (Emu3) and vector-quantized (VQ) representations, secret messages or images are encoded into natural-looking cover images and texts.

For comprehensive architectural details, reproduction instructions, experiment tables, and ongoing engineering notes, refer to **[`HANDOFF.md`](file:///home/sn634/NeuralSteganography/HANDOFF.md)**.

---

## Key Features

1. **Text-Mode Steganography**:
   - Autoregressive arithmetic coding using modern causal language models.
   - Lossless encode/decode round-trip under arbitrary contexts.

2. **Multimodal / Image Steganography**:
   - Two-stage pipeline: Secret image $\leftrightarrow$ Bitstream $\leftrightarrow$ Cover image tokens.
   - Structured image generation under discrete spatial layouts.

3. **Pixel-Level Token Recovery Optimization**:
   - Solves the VQ-VAE inversion gap where saved image pixels fail to re-tokenize to their exact original discrete codes.
   - Utilizes straight-through estimator (STE) quantization modeling, focused gradient updates on mismatched tokens, and single-frame spatial encoding.

---

## Repository Structure

```
NeuralSteganography/
├── README.md                      # Project overview
├── HANDOFF.md                     # Comprehensive technical documentation & reproduction guide
├── environment.yml                # Conda environment definition (emu3_env)
├── requirements.txt               # Key pip dependencies
│
├── src/                           # Core reusable library
│   ├── arithmetic.py              # Text-mode arithmetic coding encoder/decoder
│   ├── arithmetic_image.py        # Image-mode arithmetic coding (spatial masks & layout)
│   ├── arithmetic_image_to_bits.py# Bits ↔ VQ image token conversion routines
│   ├── optimize_pixels.py         # STE pixel optimization for exact VQ token recovery
│   └── utils.py                   # Model loaders, token/bit conversions, image I/O
│
├── scripts/                       # Runnable entry points
│   ├── run_text_steganography.py  # Text-only steganography demo
│   ├── run_image_steganography.py # Full image steganography pipeline
│   ├── run_image_generation.py    # Emu3 image generation & reconstruction sanity tests
│   ├── run_pixel_optimization.py  # Pixel optimization (Setup 1 vs Setup 2)
│   ├── run_robustness_eval.py     # Multi-seed robustness evaluation across images
│   └── run_robustness.sh          # SLURM batch job submission script
│
├── test_images/                   # Input images for benchmarking (256x256, 360x360, 720x720)
├── output_images/                 # Output directory for generated images
├── robustness_results/            # Output directory for eval metrics and plots
├── results_image_robust.txt       # Recorded experimental results
└── archive/                       # Historical exploratory tests and upstream baselines
```

---

## Quick Start

### 1. Environment Setup
```bash
conda env create -f environment.yml
conda activate emu3_env
```
Or via pip:
```bash
pip install -r requirements.txt
```

### 2. Run Text Steganography
```bash
python scripts/run_text_steganography.py
```

### 3. Run Pixel Optimization Robustness Evaluation
```bash
python scripts/run_robustness_eval.py
```

See [`HANDOFF.md`](file:///home/sn634/NeuralSteganography/HANDOFF.md) for detailed execution commands, SLURM templates, and open research directions.