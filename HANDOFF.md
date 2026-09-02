# Neural Steganography: Project Hand-Off & Reproduction Guide

**Author**: Sabrina Ning  
**Project**: Multimodal Neural Steganography (STEGASURAS & Emu3 Extensions)  
**Date**: August 2026

---

## 1. Project Overview

### 1.1 Goal
The objective of this project is to extend **neural linguistic steganography** ([Ziegler et al., 2019](https://arxiv.org/abs/1909.01496)) to multimodal generative models. Rather than concealing secret data solely within natural language, this project investigates hiding arbitrary payloads (secret text or secret images) inside **autoregressively generated cover images** using modern multimodal foundation models (primarily [BAAI/Emu3-Gen-hf](https://huggingface.co/BAAI/Emu3-Gen-hf)).

### 1.2 Key Insight
Autoregressive generative models define a conditional probability distribution $P(x_t \mid x_{<t})$ over a discrete vocabulary at every step. **Arithmetic Coding (AC)** allows a sender to encode an arbitrary bitstream into tokens sampled from this exact distribution, producing outputs that are statistically indistinguishable from unconstrained generation (near-zero KL divergence).

In image generation models like Emu3, the image is represented as a 2D grid of discrete **Vector-Quantized (VQ)** tokens. By applying arithmetic coding across the visual token generation sequence, we can embed secret information into the visual tokens of a generated cover image.

### 1.3 Core Bottleneck & Solution
When cover image tokens are decoded into raw RGB pixels, saved as standard image formats (e.g., lossless PNG uint8), and later re-encoded by the receiver's VQ encoder, **subtle pixel-level rounding errors cross VQ decision boundaries**. Even a single mismatched VQ token corrupts subsequent arithmetic coding decoding.

To solve this, we developed a **pixel-level optimization procedure (`src/optimize_pixels.py`)**:
- Uses a **Straight-Through Estimator (STE)** to model uint8 PNG quantization during gradient descent.
- Runs an optimized **single-frame spatial encoder** (bypassing 4-frame temporal video convolutions).
- Employs **focused loss** targeting only mismatched tokens once $\ge 99\%$ accuracy is achieved.
- Applies **targeted noise restarts** on spatial regions where tokens stall.

**Result (Setup 2 Benchmark)**: Achieved **100% exact token recovery (8190/8190 tokens, 10/10 runs)** on 720×720 images, with high visual fidelity.

### 1.4 Current Status
- **Text-mode steganography**: Fully functional and verified with GPT-2 / Qwen.
- **Pixel optimization for token recovery (Setup 2)**: Fully solved and benchmarked across multiple seeds and resolutions.
- **End-to-end image steganography**: Pipeline architecture is built (`scripts/run_image_steganography.py`); open challenge remains around low-probability tail tokens and vocabulary constraints in autoregressive image generation (detailed in Section 8).

---

## 2. Pipeline Architecture & Data Flow

### 2.1 Encoding Flow (Sender)

```
[ 1. Secret Payload Preparation ]
  +--------------------+          +--------------------+          +---------------------+
  |    Secret Image    | -------> |  Secret VQ Tokens  | -------> | Uniform Secret Bits |
  +--------------------+  encode  +--------------------+  decode  +---------------------+
                         (VQ-VAE)                         (AC+LM)            |
                                                                             |
[ 2. Cover Generation & Embedding ]                                          |
  +--------------------+      +----------------------+                       |
  | Prompt Context     | ---> | Emu3 Autoregressive  |                       |
  | ("a puppy", etc.)  |      | Model Logits P(x_t)  |                       |
  +--------------------+      +----------------------+                       |
                                         |                                   |
                                         v                                   v
                              +----------------------------------------------------+
                              |   Cover VQ Tokens (via AC Encode + Spatial Mask)   |
                              +----------------------------------------------------+
                                                        |
                                                        v decode (VQ-VAE)
                              +----------------------------------------------------+
                              |             Raw Decoded Cover Pixels               |
                              +----------------------------------------------------+
                                                        |
[ 3. Pixel Robustification ]                            v
                              +----------------------------------------------------+
                              |  Pixel Optimizer (STE Quantization + Focused Loss) |
                              +----------------------------------------------------+
                                                        |
                                                        v save
                              +----------------------------------------------------+
                              |   Final Cover Image PNG (100% Token Recoverable)   |
                              +----------------------------------------------------+
```

### 2.2 Decoding Flow (Receiver)

```
[ 1. Token Recovery ]
  +----------------------------------------------------+
  |              Received Cover Image PNG              |
  +----------------------------------------------------+
                             |
                             v encode (VQ-VAE)
  +----------------------------------------------------+
  |               Recovered Cover Tokens               |
  +----------------------------------------------------+
                             |
[ 2. Bitstream Extraction ]  |
                             v decode (AC + Cover Prompt Context)
  +----------------------------------------------------+
  |               Recovered Secret Bits                |
  +----------------------------------------------------+
                             |
[ 3. Secret Reconstruction ] |
                             v encode (AC + Secret Model)
  +----------------------------------------------------+
  |                 Secret VQ Tokens                   |
  +----------------------------------------------------+
                             |
                             v decode (VQ-VAE)
  +----------------------------------------------------+
  |               Recovered Secret Image               |
  +----------------------------------------------------+
```

---

## 3. Repository Map

```
NeuralSteganography/
├── README.md                          # Project introduction and quick-start guide
├── HANDOFF.md                         # This comprehensive hand-off & reproduction document
├── environment.yml                    # Conda environment definition (emu3_env)
├── requirements.txt                   # Minimal pip dependencies
├── .gitignore                         # Configured to track test images, ignore large outputs
├── results_image_robust.txt           # Recorded experimental benchmark results
│
├── src/                               # Core library code
│   ├── __init__.py                    # Package marker
│   ├── arithmetic.py                  # Text-mode AC encoder & decoder (GPT-2, Qwen, Emu3)
│   ├── arithmetic_image.py            # Image-mode AC with Emu3 spatial layout & masks
│   ├── arithmetic_image_to_bits.py    # Bits ↔ VQ image token conversion routines
│   ├── optimize_pixels.py             # STE pixel optimization for exact VQ token recovery
│   └── utils.py                       # Model loaders, token/bit conversions, image I/O
│
├── scripts/                           # Executable entry points
│   ├── run_text_steganography.py      # Text steganography demo (GPT-2 baseline)
│   ├── run_image_steganography.py     # End-to-end image-in-image steganography pipeline
│   ├── run_image_generation.py        # Emu3 generation and VQ reconstruction tests
│   ├── run_pixel_optimization.py      # Setup 1 vs Setup 2 pixel optimization comparison
│   ├── run_robustness_eval.py         # Multi-seed robustness benchmark across test images
│   └── run_robustness.sh              # SLURM sbatch script for cluster execution
│
├── test_images/                       # Benchmark test images (tracked in git, ~2.4 MB)
│   ├── cat_256x256.png, cat_360x360.png, cat_512x512.png, cat_720x720.png
│   ├── dog_256x256.png, dog_360x360.png, dog_512x512.png, dog_720x720.png
│   ├── nature_256x256.png, nature_360x360.png, nature_512x512.png
│   └── yosemite_720x720.png
│
├── output_images/                     # Pipeline output artifacts (gitignored)
│   └── README.md                      # Directory placeholder
│
├── robustness_results/                # Multi-seed evaluation outputs & plots (gitignored)
│   └── README.md                      # Directory placeholder
│
└── archive/                           # Historical exploratory tests and upstream baselines
    ├── README.md                      # Index and descriptions of archived experiments
    ├── emu3_source_model_tests.py     # was test_2.py — original non-HF Emu3 sampling
    ├── entropy_stress_test_emu3.py    # was test_3.py — Emu3-Chat entropy dynamics
    ├── entropy_stress_test_gpt2.py    # was test_4.py — GPT-2 entropy dynamics
    ├── cache_truncation_gpt2xl.py     # was test_5.py — KV-cache truncation experiments
    ├── long_generation_olmo.py        # was test_6.py — OLMo-7B long generation
    ├── text_ac_roundtrip_test.py      # was test_arithmetic.py — text AC encode/decode test
    ├── image_text_ac_test.py          # was test_arithmetic_2.py — initial image+text pipeline
    ├── token_comparison.py            # was test_load_image.py — VQ token comparator
    ├── unit_tests.py                  # Model exploration suite
    ├── run_arithmetic.py              # Multi-model text steganography tests
    ├── block_baseline.py              # Original binning steganography baseline
    ├── huffman.py                     # Original Huffman coding implementation
    ├── huffman_baseline.py            # Original Huffman steganography baseline
    ├── sample.py                      # Unconstrained sampling baseline
    └── vocab/                         # Tokenizer vocabulary CSVs
```

---

## 4. Environment Setup

### 4.1 Conda Environment (Recommended)
The full environment is defined in `environment.yml`:

```bash
# Create and activate environment
conda env create -f environment.yml
conda activate emu3_env
```

### 4.2 Pip Installation (Alternative)
If setting up in a pre-existing PyTorch environment with CUDA 12:

```bash
pip install -r requirements.txt
```

Key packages and tested versions:
- `torch >= 2.7.1` (CUDA 12.6)
- `transformers >= 4.52.4`
- `accelerate >= 1.8.1`
- `einops`, `bitarray`, `pillow`, `matplotlib`, `numpy`, `scipy`
- `scikit-image`, `scikit-learn`, `lpips`, `torchmetrics`

---

## 5. How to Reproduce Results

Make sure your current working directory is the repository root:
```bash
cd /home/<YOUR_USERNAME>/NeuralSteganography
```

### 5.1 Token Recovery Optimization (Setup 2 Benchmark)
This reproduces the core confirmed empirical result: optimizing decoded pixels to achieve 100% token recovery through PNG quantization across multiple seeds and images.

```bash
python scripts/run_robustness_eval.py
```
- **What it does**: Runs Setup 2 pixel optimization on test images across 5 seeds each, measures token match percentage before and after PNG round-trip, computes PSNR, SSIM, and LPIPS, and writes plots and difference maps to `robustness_results/`.

### 5.2 Text-Only Steganography Round-Trip
Verifies lossless arithmetic coding embedding and decoding into natural text:

```bash
python scripts/run_text_steganography.py
```
- **What it does**: Uses GPT-2 to arithmetic-code a secret Wikipedia passage into cover text starting from a historical context prompt. Recovers the exact original message bits and decodes the plaintext.

### 5.3 Emu3 Image Generation & VQ Reconstruction Sanity Test
Verifies that Emu3 generates image tokens, decodes to pixels, and re-tokenizes:

```bash
python scripts/run_image_generation.py
```
- **What it does**: Generates a 720×720 image from the prompt `"a kitten"`, saves `output_images/generated.png`, and inspects token re-encoding discrepancy.

### 5.4 Setup 1 vs. Setup 2 Pixel Optimization Comparison
```bash
python scripts/run_pixel_optimization.py
```
- **What it does**: Analyzes the VQ embedding distance distributions between Setup 1 (optimizing toward original AR tokens) and Setup 2 (optimizing toward re-encoded self-consistent tokens), generating histogram diagnostics in `output_images/initial_distance_hist.png`.

### 5.5 Full End-to-End Image Steganography Pipeline
```bash
python scripts/run_image_steganography.py
```
- **What it does**: Runs the full two-stage pipeline: encodes `test_images/yosemite_720x720.png` into secret bits, embeds those bits into cover image tokens under the prompt `"a puppy"`, optimizes cover pixels, saves the PNG, and attempts full decoding.

---

## 6. Key Technical Implementation Details

### 6.1 Arithmetic Coding (AC) Mechanics
In **image mode** (`src/arithmetic_image.py`), spatial layout rules are enforced:
- For a $90 \times 90$ token image ($720 \times 720$ pixels):
  - Tokens $1 \dots 90$ are visual tokens carrying payload bits.
  - Token $91$ is forced to be the deterministic EOL marker (`<|extra_200|>`), carrying 0 payload bits.
  - At the end of the grid, deterministic EOF, EOI, and EOS tokens are appended.

### 6.2 Straight-Through Estimator (STE) for Pixel Quantization
Standard gradient descent on continuous pixel values cannot anticipate the floor/rounding operation when saving an image to an 8-bit PNG:
$$\text{pixel}_{uint8} = \text{clamp}\left(\lfloor (pv \times 0.5 + 0.5) \times 255 \rfloor, 0, 255\right)$$

In `src/optimize_pixels.py`, we implement an STE in double precision:
```python
def _quantize_ste(pv):
    pv_255 = (pv.double() * 0.5 + 0.5) * 255.0
    pv_floored = pv_255.floor().clamp(0, 255)
    pv_ste = pv_255 + (pv_floored - pv_255).detach()
    pv_rescaled = pv_ste * image_processor.rescale_factor
    return ((pv_rescaled - 0.5) / 0.5).float()
```
The forward pass uses the exact quantized values, while backward gradients pass straight through to continuous pixel parameters.

### 6.3 Single-Frame Spatial Encoder Bypass
Emu3's VQ encoder is designed for video and expects temporal inputs with factor $T=4$. In naive execution, passing a still image duplicates it across 4 frames, quadrupling memory and compute during backward passes.

`_spatial_encode(pv)` extracts only the 2D spatial ResNet and self-attention blocks for a single frame, expanding temporally only at the final bottleneck time-convolution. This accelerates each optimization step by $\sim 3.2\times$.

### 6.4 Focused Loss & Adaptive Learning Rate Decay
When $99\%$ of tokens match the ground truth, standard global MSE over the entire latent feature map causes gradient interference, flipping previously correct tokens:
- **Focused Loss**: Restricts the MSE loss exclusively to mismatched token positions:
  $$\mathcal{L} = \frac{1}{|M|} \sum_{i \in M} \|\mathbf{z}_i - \mathbf{e}_{target, i}\|^2$$
- **Adaptive LR Decay**: Scales down the learning rate as errors decrease:
  $$\eta_{eff} = \eta \cdot \max\left(\frac{N_{mis}}{N_{total}}, 0.05\right)$$
- **Targeted Random Restart**: If no progress is observed for `patience=100` iterations, localized Gaussian noise is injected only into the image regions corresponding to the mismatched tokens.

### 6.5 Experimental Setups: Setup 1 vs. Setup 2

In `scripts/run_pixel_optimization.py`, two experimental configurations were formulated to isolate and analyze the source of VQ token quantization errors:

| Feature | **Setup 1 (Real Steganography Scenario)** | **Setup 2 (Self-Consistent Diagnostic Control)** |
|---|---|---|
| **Starting Pixels** | Raw decoded pixels saved to PNG uint8, then reloaded from disk ($\mathbf{P}_{reload}$). | Decoded from tokens that were re-encoded from PNG ($\mathbf{P}_{fresh} = \text{Decode}(\text{Encode}(\mathbf{P}_{reload}))$). |
| **Optimization Target** | Original autoregressively sampled tokens ($\mathbf{T}_{AR}$ / `ground_truth_tokens.pt`). | The re-encoded tokens ($\mathbf{T}_{re\text{-}enc} = \text{Encode}(\mathbf{P}_{reload})$). |
| **Initial Gap** | **Large**: $\mathbf{T}_{AR}$ and $\mathbf{P}_{reload}$ have slight Voronoi mismatches due to decoder imperfections and uint8 PNG quantization. | **Near Zero**: $\mathbf{T}_{re\text{-}enc}$ directly produced $\mathbf{P}_{fresh}$ in the same decode pass, making targets self-consistent. |
| **Purpose** | **The actual steganography problem**: Proves whether arbitrary AR-generated tokens carrying secret payload bits can be forced to survive PNG round-trip through pixel perturbation. | **Diagnostic / Sanity check**: Verifies optimizer mechanics and serves as an empirical baseline to confirm whether failures in Setup 1 stem from optimizer defects or token unreachability. |

**Key Takeaway**: Setup 2 is the main pipeline used in the existing benchmarks, achieving **100% token recovery (10/10 runs on 720×720 images)**.

---

## 7. Experimental Results Summary

The following results are recorded from benchmark runs documented in `results_image_robust.txt`.

### 7.1 Multi-Seed Evaluation at 720×720 (Updated Algorithm)
- **Target Image Size**: $720 \times 720$ (8100 visual tokens + 90 EOL tokens = 8190 tokens total)
- **Optimizer**: Adam ($\text{lr}=5 \times 10^{-4}$, patience=100, max iterations=1000)

| Image | Seed | Status | Optimized Tokens | Verified Tokens | Iterations | PSNR (dB) | SSIM | LPIPS | Time (s) |
|---|---|---|---|---|---|---|---|---|---|
| `cat_720x720.png` | 0 | **OK** | 8100/8100 | 8190/8190 | 390 | 33.68 | 0.9399 | 0.0255 | 399.5 |
| `cat_720x720.png` | 1 | **OK** | 8100/8100 | 8190/8190 | 358 | 33.68 | 0.9403 | 0.0255 | 367.5 |
| `cat_720x720.png` | 2 | **OK** | 8100/8100 | 8190/8190 | 295 | 33.71 | 0.9424 | 0.0237 | 303.2 |
| `cat_720x720.png` | 3 | **OK** | 8100/8100 | 8190/8190 | 387 | 33.68 | 0.9400 | 0.0258 | 397.5 |
| `cat_720x720.png` | 4 | **OK** | 8100/8100 | 8190/8190 | 348 | 33.67 | 0.9402 | 0.0255 | 357.2 |
| `dog_720x720.png` | 0 | **OK** | 8100/8100 | 8190/8190 | 506 | 34.38 | 0.9291 | 0.0517 | 519.8 |
| `dog_720x720.png` | 1 | **OK** | 8100/8100 | 8190/8190 | 397 | 34.52 | 0.9350 | 0.0426 | 407.3 |
| `dog_720x720.png` | 2 | **OK** | 8100/8100 | 8190/8190 | 446 | 34.47 | 0.9326 | 0.0474 | 457.7 |
| `dog_720x720.png` | 3 | **OK** | 8100/8100 | 8190/8190 | 490 | 34.44 | 0.9317 | 0.0480 | 502.9 |
| `dog_720x720.png` | 4 | **OK** | 8100/8100 | 8190/8190 | 529 | 34.45 | 0.9318 | 0.0477 | 543.2 |
| **Average** | — | **100% OK** | **8100/8100** | **8190/8190** | **414.6** | **34.07** | **0.9363** | **0.0363** | **425.6** |

### 7.2 Summary Across Smaller Resolutions
- **256×256 images** (1024 tokens): 15/15 runs OK (100% recovery), avg 125 iters ($\sim 45\text{s}$).
- **360×360 images** (2025 tokens): 15/15 runs OK (100% recovery), avg 180 iters ($\sim 125\text{s}$).

---

## 8. Known Issues & Open Problems

### 8.1 "Unreachable Tokens" in Autoregressive Sampling
During autoregressive cover image generation, Emu3 samples tokens from a large vocabulary (151k tokens total, with 8192 visual codebook indices). In unconstrained generation, the model occasionally samples visual tokens that lie in extremely thin or unstable Voronoi cells in VQ latent space. 
- While Setup 2 optimizes pixels toward tokens derived from existing images (where 100% token recovery was achieved), end-to-end embedding requires arbitrary autoregressively sampled tokens in the sequence to be reachable and encodable through the AC probability window.

### 8.2 Optimization Runtimes
Optimizing 720×720 images takes $\sim 6\text{–}9$ minutes per image. While acceptable for asynchronous steganographic communication, faster convergence or better initialization is desirable.

---

## 9. K-Means & Margin Loss Optimization

We investigated a **K-means clustering + margin loss** technique that achieved **15/20 token/image recovery** under constrained settings. Below is the conceptual documentation.

### 9.1 The Concept
1. **Codebook Geometry**: The Emu3 VQ codebook $\mathbf{E} \in \mathbb{R}^{8192 \times 4}$ contains 8192 embeddings in 4-dimensional latent space. Many embeddings are tightly clustered, leaving very small decision margins.
2. **K-Means Clustering**: Cluster the 8192 codebook vectors into $K$ well-separated cluster centers (e.g., $K=1024$ or $2048$).
3. **Constrained Generation Vocabulary**: Restrict autoregressive sampling to tokens nearest to cluster centroids. This prevents the model from choosing ambiguous boundary tokens.
4. **Margin Loss**: In the pixel optimizer, augment the MSE loss with a margin penalty that explicitly pushes the encoded latent $\mathbf{z}$ away from the second-nearest codebook entry $\mathbf{e}_{second}$:
   $$\mathcal{L}_{margin} = \max\left(0, \|\mathbf{z} - \mathbf{e}_{target}\|^2 - \|\mathbf{z} - \mathbf{e}_{second}\|^2 + m\right)$$
   where $m > 0$ is a safety margin.

### 9.2 Implementation Sketch (Pseudocode)

```python
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans

def compute_codebook_clusters(vqmodel, n_clusters=2048):
    """Cluster VQ codebook into well-separated clusters."""
    embeddings = vqmodel.quantize.embedding.weight.detach().cpu().numpy()  # [8192, 4]
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(embeddings)
    cluster_centers = torch.tensor(kmeans.cluster_centers_, device="cuda")
    # For each cluster, find the closest actual codebook token id
    dists = torch.cdist(cluster_centers, torch.tensor(embeddings, device="cuda"))
    allowed_token_ids = dists.argmin(dim=1).unique()
    return allowed_token_ids

def margin_loss(conv_flat, target_indices, vqmodel, margin=0.1):
    """
    Penalize latents that are too close to competing codebook entries.
    conv_flat: [N, 4]
    target_indices: [N]
    """
    codebook = vqmodel.quantize.embedding.weight  # [8192, 4]
    target_emb = codebook[target_indices]         # [N, 4]
    
    # Distance to target
    d_target = (conv_flat - target_emb).pow(2).sum(-1)  # [N]
    
    # Distance to all entries
    all_dists = torch.cdist(conv_flat, codebook).pow(2)  # [N, 8192]
    # Mask out the target entry to find the closest competitor
    all_dists.scatter_(1, target_indices.unsqueeze(1), float('inf'))
    d_second, _ = all_dists.min(dim=1)  # [N]
    
    # Margin hinge loss
    loss_margin = F.relu(d_target - d_second + margin).mean()
    loss_mse = d_target.mean()
    return loss_mse + 0.5 * loss_margin
```

---

## 10. Potential Next Steps to Explore

1. **Integrate Cluster-Constrained Sampling**:
   Use `prefix_allowed_tokens_fn` in HuggingFace generation to restrict Emu3 cover generation to the high-margin codebook subset identified via K-means.
2. **Combine Margin Loss with STE Optimization**:
   Incorporate the margin loss function above into `src/optimize_pixels.py` to accelerate convergence on difficult regions.
3. **Adaptive Precision for Tail Tokens**:
   Implement a dynamic precision fallback in `src/arithmetic_image_to_bits.py` when encountering tokens with log-probabilities below $10^{-10}$.
4. **Diffusion / Continuous Steganography Comparison**:
   Compare discrete VQ steganography against continuous latent diffusion steganography.
