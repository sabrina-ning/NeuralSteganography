"""
Run Setup 1 of test_image_robust.py exactly 5 times with different seeds.

Setup 1: reload noisy pixel values from output_images/generated.png and optimize
toward the AR-sampled tokens stored in output_images/ground_truth_tokens.pt.
"""

import time
import os
from PIL import Image
from transformers import AutoProcessor, Emu3ForConditionalGeneration
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import lpips
from torchmetrics.image import StructuralSimilarityIndexMeasure

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
model_id = "BAAI/Emu3-Gen-hf"
device = "cuda" if torch.cuda.is_available() else "cpu"
N_RUNS = 5
MAX_ITERS = 1000
PATIENCE = 100
LR = 5e-4
HEIGHT, WIDTH = 720, 720
OUTPUT_DIR = "robustness_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Load model once
# ---------------------------------------------------------------------------
print(f"Loading model {model_id} ...")
model = Emu3ForConditionalGeneration.from_pretrained(
    model_id, dtype=torch.bfloat16, device_map=device)
base_model = model.model
vqmodel = base_model.vqmodel.float()

processor = AutoProcessor.from_pretrained(model_id, dtype=torch.bfloat16, device_map=device)
image_processor = processor.image_processor
image_processor.do_resize = False
downsample_ratio = image_processor.spatial_factor

# ---------------------------------------------------------------------------
# Optimization (mirror of test_image_robust.optimize_pixels)
# ---------------------------------------------------------------------------
def optimize_pixels(noisy_pixel_values, original_tokens, height, width,
                    n_iterations=MAX_ITERS, lr=LR, patience=PATIENCE):
    bpe_grid = original_tokens.view(height // downsample_ratio, width // downsample_ratio + 1)
    vq_indices = base_model.vocabulary_mapping.convert_bpe2img(bpe_grid).flatten().to(device)
    target_embeddings = vqmodel.quantize.embedding(vq_indices).detach()

    modified = noisy_pixel_values.clone().float().requires_grad_(True)
    optimizer = torch.optim.Adam([modified], lr=lr)
    losses = []
    best_matches = 0
    best_pixels = modified.detach().clone()
    use_focused_crossed = False
    ever_matched = torch.zeros(len(vq_indices), dtype=torch.bool, device=device)

    enc = vqmodel.encoder

    def _quantize_ste(pv):
        pv_255 = (pv.double() * 0.5 + 0.5) * 255.0
        pv_floored = pv_255.floor().clamp(0, 255)
        pv_ste = pv_255 + (pv_floored - pv_255).detach()
        pv_rescaled = pv_ste * image_processor.rescale_factor
        return ((pv_rescaled - 0.5) / 0.5).float()

    def _spatial_encode(pv):
        h = enc.conv_in(pv)
        for down_level in enc.down_block.down:
            for i_block, resnet_block in enumerate(down_level.block):
                h = resnet_block(h)
                if len(down_level.attn) > 0:
                    residual = h
                    h = down_level.attn_norms[i_block](h)
                    bs, ch, ht, wd = h.shape
                    h = h.view(bs, ch, ht * wd).transpose(1, 2)
                    h = down_level.attn[i_block](h)[0]
                    h = h.reshape(bs, ht, wd, ch).permute(0, 3, 1, 2)
                    h = residual + h
            if hasattr(down_level, 'downsample'):
                h = down_level.downsample(h)
        h = enc.middle_block(h)
        h = enc.norm_out(h)
        h = h * torch.sigmoid(h)
        return enc.conv_out(h)

    for i in range(n_iterations):
        optimizer.zero_grad()

        hidden = _spatial_encode(_quantize_ste(modified))
        hidden = hidden.unsqueeze(2).expand(
            -1, -1, vqmodel.config.temporal_downsample_factor, -1, -1)
        for t_conv in enc.time_conv:
            hidden = t_conv(hidden)
            hidden = hidden * torch.sigmoid(hidden)
        for t_res in enc.time_res_stack:
            hidden = t_res(hidden)
        hidden = hidden.permute(0, 2, 1, 3, 4)

        conv = hidden.permute(0, 2, 1, 3, 4)
        conv = vqmodel.quant_conv(conv)
        conv = conv.permute(0, 2, 1, 3, 4)

        conv_flat = conv.squeeze(0).squeeze(0).permute(1, 2, 0).reshape(-1, 4)

        with torch.no_grad():
            current_indices = vqmodel.quantize(conv.detach()).flatten()
            matches = int((current_indices == vq_indices).sum().item())
            ever_matched |= (current_indices == vq_indices)

        if matches == len(vq_indices):
            print(f"    iter {i}: all tokens matched")
            best_pixels = modified.detach().clone()
            best_matches = matches
            break

        if matches > best_matches:
            best_matches = matches
            best_pixels = modified.detach().clone()

        mismatched = current_indices != vq_indices
        match_ratio = matches / len(vq_indices)
        use_focused = match_ratio >= 0.99
        if use_focused:
            if not use_focused_crossed:
                use_focused_crossed = True
                print(f"    iter {i}: focused loss engaged (lr decays with n_mis)")
            n_mis = mismatched.sum().item()
            decay = max(n_mis / len(vq_indices), 0.05)
            for pg in optimizer.param_groups:
                pg['lr'] = lr * decay
            loss = F.mse_loss(conv_flat[mismatched], target_embeddings[mismatched])
        else:
            for pg in optimizer.param_groups:
                pg['lr'] = lr
            loss = F.mse_loss(conv_flat, target_embeddings)

        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if i % 100 == 0:
            print(f"    iter {i}: tokens {matches}/{len(vq_indices)}")

    ever_count = int(ever_matched.sum().item())
    never_count = len(vq_indices) - ever_count
    print(f"    reachability: {ever_count}/{len(vq_indices)} matched at some point "
          f"({never_count} never matched)")

    return (i + 1 if matches == len(vq_indices) else n_iterations,
            best_matches, len(vq_indices), losses, best_pixels)


# ---------------------------------------------------------------------------
# Fixed inputs (same across all 5 runs)
# ---------------------------------------------------------------------------
ground_truth_tokens = torch.load("output_images/ground_truth_tokens.pt").to(device)
reloaded_image = Image.open("output_images/generated.png")
pre = image_processor.preprocess(reloaded_image, do_normalize=True, return_tensors="pt")
noisy_pixel_values = pre["pixel_values"].float().to(device)

# Perceptual metrics
ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0)
lpips_metric = lpips.LPIPS(net='alex')
lpips_metric.eval()

# ---------------------------------------------------------------------------
# Run Setup 1 five times with different seeds
# ---------------------------------------------------------------------------
results = []

for run in range(N_RUNS):
    seed = run
    print(f"\n{'='*60}")
    print(f"[{run + 1}/{N_RUNS}] Setup 1  seed={seed}")
    print(f"{'='*60}")

    torch.manual_seed(seed)

    t0 = time.time()
    iters, matched, total, losses, recovered_pixels = optimize_pixels(
        noisy_pixel_values, ground_truth_tokens, HEIGHT, WIDTH)
    elapsed = time.time() - t0

    tag = f"setup1_seed{seed}"

    if losses:
        plt.figure()
        plt.plot(losses)
        plt.xlabel("iteration")
        plt.ylabel("loss")
        plt.title(f"{tag} — opt {matched}/{total}")
        plt.savefig(os.path.join(OUTPUT_DIR, f"loss_{tag}.png"))
        plt.close()

    recovered_image = image_processor.postprocess(
        recovered_pixels.float(), return_tensors="PIL.Image.Image")["pixel_values"][0]
    recovered_image.save(os.path.join(OUTPUT_DIR, f"recovered_{tag}.png"))

    reloaded = image_processor.preprocess(recovered_image, do_normalize=True, return_tensors="pt")
    reloaded_pixels = reloaded["pixel_values"].float().to(device)
    reloaded_sizes = reloaded["image_sizes"].to(device)
    with torch.no_grad():
        reloaded_tokens = base_model.get_image_tokens(reloaded_pixels, reloaded_sizes).to(device)
        verified_match = int((ground_truth_tokens == reloaded_tokens).sum().item())
        verified_total = len(ground_truth_tokens)

    status = "OK" if verified_match == verified_total else "STUCK"

    orig_t = torch.from_numpy(np.array(reloaded_image)).float().div(255).permute(2, 0, 1).unsqueeze(0)
    recv_t = torch.from_numpy(np.array(recovered_image)).float().div(255).permute(2, 0, 1).unsqueeze(0)
    mse_val = F.mse_loss(recv_t, orig_t)
    psnr_val = (10 * torch.log10(1.0 / mse_val)).item() if mse_val > 0 else float('inf')
    ssim_val = ssim_metric(recv_t, orig_t).item()
    with torch.no_grad():
        lpips_val = lpips_metric(2 * orig_t - 1, 2 * recv_t - 1).item()

    results.append({
        "seed": seed,
        "status": status,
        "iters": iters,
        "opt_matched": matched,
        "verified_matched": verified_match,
        "total": verified_total,
        "time_s": elapsed,
        "psnr": psnr_val,
        "ssim": ssim_val,
        "lpips": lpips_val,
    })
    print(f"  -> {status}: opt={matched}/{total}, verified={verified_match}/{verified_total} "
          f"in {iters} iters ({elapsed:.1f}s)")
    print(f"     PSNR={psnr_val:.2f}dB  SSIM={ssim_val:.4f}  LPIPS={lpips_val:.4f}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print("SUMMARY (Setup 1 x 5)")
print(f"{'='*60}")
print(f"{'Seed':>4} {'Status':<6} {'Opt':>12} {'Verified':>12} {'Iters':>6} "
      f"{'PSNR':>8} {'SSIM':>7} {'LPIPS':>7} {'Time':>7}")
print("-" * 80)
for r in results:
    print(f"{r['seed']:>4} {r['status']:<6} "
          f"{r['opt_matched']:>5}/{r['total']:<5} "
          f"{r['verified_matched']:>5}/{r['total']:<5} {r['iters']:>6} "
          f"{r['psnr']:>7.2f} {r['ssim']:>7.4f} {r['lpips']:>7.4f} {r['time_s']:>6.1f}s")

avg_psnr = np.mean([r["psnr"] for r in results if r["psnr"] != float('inf')])
avg_ssim = np.mean([r["ssim"] for r in results])
avg_lpips = np.mean([r["lpips"] for r in results])
print(f"\nAverage metrics:  PSNR={avg_psnr:.2f}dB  SSIM={avg_ssim:.4f}  LPIPS={avg_lpips:.4f}")

stuck = [r for r in results if r["status"] == "STUCK"]
print(f"\n{len(results)} runs total, {len(stuck)} STUCK, {len(results) - len(stuck)} OK")
print(f"\nOutputs saved to {OUTPUT_DIR}/")
