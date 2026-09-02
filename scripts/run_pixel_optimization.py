from PIL import Image
from transformers import AutoProcessor, Emu3ForConditionalGeneration
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from scripts.run_image_generation import manual_generate

# Global constants
model_id = "BAAI/Emu3-Gen-hf"
prompt_str = "a kitten"
seed = 42
torch.manual_seed(seed)
device = "cuda" if torch.cuda.is_available() else "cpu"

model = Emu3ForConditionalGeneration.from_pretrained(
  model_id,
  dtype=torch.bfloat16,
  device_map=device)
base_model = model.model
print("base_model type:", type(base_model))
vqmodel = base_model.vqmodel.float()  # float32 for precision

processor = AutoProcessor.from_pretrained(
  model_id,
  dtype=torch.bfloat16,
  device_map=device)
image_processor = processor.image_processor
image_processor.do_resize = False
downsample_ratio = image_processor.spatial_factor
height, width = 720, 720

# inputs = processor(
#     text=[prompt_str],
#     padding=True,
#     return_tensors="pt",
#     return_for_image_generation=True,
# ).to(model.device)
#
# with torch.no_grad():
#     generated_tokens = manual_generate(model, processor, inputs, seed)[0].to(device)
#     ground_truth_tokens = generated_tokens[:-3]  # strip trailing eof/eoi/eos specials
#     print("ground truth tokens:", ground_truth_tokens)
#     torch.save(ground_truth_tokens.cpu(), "output_images/ground_truth_tokens.pt")
#
#     # Save and visualize the generated image
#     gen_pixels = base_model.decode_image_tokens(
#         generated_tokens.unsqueeze(0),
#         height=(height // downsample_ratio),
#         width=(width // downsample_ratio))
#     gen_image = image_processor.postprocess(
#         gen_pixels.float(), return_tensors="PIL.Image.Image")["pixel_values"][0]
#     gen_image.save("output_images/generated.png")
#     print("generated image saved to output_images/generated.png")

ground_truth_tokens = torch.load("output_images/ground_truth_tokens.pt").to(device)
print("loaded ground truth tokens:", ground_truth_tokens)
torch.cuda.empty_cache()

def _quantize_ste(pv):
    """
    Simulate PNG uint8 round-trip on pixel values `pv` with straight-through estimator.
    Uses float64 (double) precision to perfectly match NumPy's behavior in `postprocess`.
    """
    pv_255 = (pv.double() * 0.5 + 0.5) * 255.0 # cast to float64, denormalize to [0, 255]
    pv_floored = pv_255.floor().clamp(0, 255) # match uint8 truncation
    pv_ste = pv_255 + (pv_floored - pv_255).detach()
    pv_rescaled = pv_ste * image_processor.rescale_factor # rescale to [0, 1]
    return ((pv_rescaled - 0.5) / 0.5).float() # renormalize, cast back to float32

def _spatial_encode(pv):
    """
    Run spatial encoder on only one frame. Avoids running on temporal batch of 4,
    since all frames are identical for a still image.
    """
    enc = vqmodel.encoder
    h = enc.conv_in(pv) # [1, ch=3, H, W]
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
    return enc.conv_out(h)  # [1, lat_ch=4, H/8, W/8]

def encode_to_latents(pixel_values, apply_ste=True):
    """
    Run the full VQ-VAE encoder forward (spatial + temporal + quant_conv) on
    `pixel_values`. Returns (conv, conv_flat) where conv is [1, 1, lat_ch, H', W']
    and conv_flat is [n_tokens, lat_ch] for direct codebook comparison.
    """
    enc = vqmodel.encoder
    pv = _quantize_ste(pixel_values) if apply_ste else pixel_values
    hidden = _spatial_encode(pv)
    hidden = hidden.unsqueeze(2).expand(-1, -1, vqmodel.config.temporal_downsample_factor, -1, -1)
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
    return conv, conv_flat

def bpe_to_vq_indices(original_tokens, height, width):
    """Convert a flat BPE-with-EOL token tensor to flat img-space VQ indices."""
    bpe_grid = original_tokens.view(height // downsample_ratio, width // downsample_ratio + 1)
    return base_model.vocabulary_mapping.convert_bpe2img(bpe_grid).flatten().to(device)

def diagnose(noisy_pixel_values, vq_indices, tag):
    """
    Single no-grad forward pass. Reports how far the encoder output sits from
    the target codebook embeddings vs. its actually-nearest codebook entries.
    Returns a dict with per-token tensors for plotting.
    """
    print(f"\n--- diagnose [{tag}] ---")
    target_embeddings = vqmodel.quantize.embedding(vq_indices).detach()

    with torch.no_grad():
        conv, conv_flat = encode_to_latents(noisy_pixel_values, apply_ste=True)
        current_indices = vqmodel.quantize(conv).flatten()
        nearest_embeddings = vqmodel.quantize.embedding(current_indices)

        dist_to_target = (conv_flat - target_embeddings).pow(2).sum(1)
        dist_to_nearest = (conv_flat - nearest_embeddings).pow(2).sum(1)
        gap = dist_to_target - dist_to_nearest  # >0 means encoder prefers a wrong entry

        n_tokens = len(vq_indices)
        initial_matches = int((current_indices == vq_indices).sum().item())
        n_pos_gap = int((gap > 0).sum().item())

        def q(t, p):
            return float(torch.quantile(t, p).item())

        print(f"[{tag}] initial matches: {initial_matches}/{n_tokens} "
              f"({100*initial_matches/n_tokens:.2f}%)")
        print(f"[{tag}] dist_to_target  mean={dist_to_target.mean().item():.6f} "
              f"median={q(dist_to_target,0.5):.6f} "
              f"p90={q(dist_to_target,0.9):.6f} "
              f"max={dist_to_target.max().item():.6f}")
        print(f"[{tag}] dist_to_nearest mean={dist_to_nearest.mean().item():.6f} "
              f"median={q(dist_to_nearest,0.5):.6f} "
              f"p90={q(dist_to_nearest,0.9):.6f} "
              f"max={dist_to_nearest.max().item():.6f}")
        print(f"[{tag}] gap (target - nearest) mean={gap.mean().item():.6f} "
              f"median={q(gap,0.5):.6f} "
              f"p90={q(gap,0.9):.6f} "
              f"max={gap.max().item():.6f}")
        print(f"[{tag}] tokens with positive gap (encoder prefers wrong): "
              f"{n_pos_gap}/{n_tokens} ({100*n_pos_gap/n_tokens:.2f}%)")

        return {
            "tag": tag,
            "dist_to_target": dist_to_target.cpu(),
            "dist_to_nearest": dist_to_nearest.cpu(),
            "gap": gap.cpu(),
            "initial_matches": initial_matches,
            "n_tokens": n_tokens,
        }

def plot_diagnostics(diagnostics, path="output_images/initial_distance_hist.png"):
    """Overlay per-token distance histograms across setups for side-by-side comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics = [("dist_to_target", "distance to target embedding"),
               ("dist_to_nearest", "distance to nearest embedding"),
               ("gap", "gap (target - nearest)")]
    for ax, (key, title) in zip(axes, metrics):
        all_vals = torch.cat([d[key] for d in diagnostics]).numpy()
        lo, hi = float(np.quantile(all_vals, 0.01)), float(np.quantile(all_vals, 0.99))
        bins = np.linspace(lo, hi, 60)
        for d in diagnostics:
            ax.hist(d[key].numpy(), bins=bins, alpha=0.5, label=d["tag"])
        ax.set_title(title)
        ax.set_xlabel(title)
        ax.set_ylabel("token count")
        ax.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"diagnostic histograms saved to {path}")

def optimize_pixels(noisy_pixel_values, vq_indices, height, width,
                    n_iterations=2000, lr=5e-4, patience=200):
    # vq_indices: flat tensor of img-space VQ codebook indices (EOL already stripped)

    target_embeddings = vqmodel.quantize.embedding(vq_indices).detach()

    modified = noisy_pixel_values.clone().float().requires_grad_(True)
    optimizer = torch.optim.Adam([modified], lr=lr)
    losses = []
    prev_matches = 0
    best_matches = 0
    best_pixels = modified.detach().clone()
    best_current_indices = None
    window_best_matches = 0  # per-restart best, for patience triggering
    iters_since_improvement = 0
    use_focused_crossed = False

    for i in range(n_iterations):
        optimizer.zero_grad()

        conv, conv_flat = encode_to_latents(modified, apply_ste=True)

        # Check how many current tokens match the original ground truth
        with torch.no_grad():
            current_indices = vqmodel.quantize(conv.detach()).flatten()
            matches = int((current_indices == vq_indices).sum().item())

        if matches == len(vq_indices):
            print(f"iter {i}: all tokens matched")
            best_pixels = modified.detach().clone()
            best_matches = matches
            break

        if matches > best_matches:
            best_matches = matches
            best_pixels = modified.detach().clone()
            best_current_indices = current_indices.clone()

        mismatched = current_indices != vq_indices
        match_ratio = matches / len(vq_indices)
        use_focused = match_ratio >= 0.99
        if use_focused:
            if not use_focused_crossed:
                use_focused_crossed = True
                print(f"iter {i}: focused loss engaged (lr decays with n_mis)")
            n_mis = mismatched.sum().item()
            decay = max(n_mis / len(vq_indices), 0.05)
            for pg in optimizer.param_groups:
                pg['lr'] = lr * decay
            loss = F.mse_loss(conv_flat[mismatched], target_embeddings[mismatched])
        else:
            for pg in optimizer.param_groups:
                pg['lr'] = lr
            loss = F.mse_loss(conv_flat, target_embeddings)
        prev_matches = matches

        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if i % 10 == 0:
            # DEBUGGING: For mismatched tokens, show distance to target vs distance to current (wrong) embedding
            with torch.no_grad():
                mismatched = current_indices != vq_indices
                if mismatched.any():
                    mis_flat = conv_flat[mismatched]
                    mis_target = target_embeddings[mismatched]
                    mis_current = vqmodel.quantize.embedding(current_indices[mismatched])
                    dist_to_target = (mis_flat - mis_target).pow(2).sum(1)
                    dist_to_current = (mis_flat - mis_current).pow(2).sum(1)
                    gap = dist_to_target - dist_to_current  # positive = closer to wrong embedding
                    print(f"iter {i}: loss {loss.item():.6f}, tokens matching: {matches}/{len(vq_indices)} | "
                          f"mismatched gap (mean={gap.mean().item():.6f}, max={gap.max().item():.6f})")
                else:
                    print(f"iter {i}: loss {loss.item():.6f}, tokens matching: {matches}/{len(vq_indices)}")

    # DEBUGGING: plot loss curve
    plt.figure()
    plt.plot(losses)
    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.title("optimization loss curve")
    plt.savefig("output_images/loss_curve.png")
    plt.close()
    print("loss curve saved to output_images/loss_curve.png")

    return best_pixels # float32, matches vqmodel precision

def verify_and_save(recovered_pixel_values, noisy_pixel_values, gt_tokens, tag):
    delta = (recovered_pixel_values - noisy_pixel_values).abs()
    print(f"[{tag}] Pixel range: [{noisy_pixel_values.min().item():.4f}, {noisy_pixel_values.max().item():.4f}]")
    print(f"[{tag}] Max pixel change: {delta.max().item():.6f}")
    print(f"[{tag}] Min pixel change: {delta.min().item():.6f}")
    print(f"[{tag}] Mean pixel change: {delta.mean().item():.6f}")

    recovered_image = image_processor.postprocess(
        recovered_pixel_values.float(), return_tensors="PIL.Image.Image")["pixel_values"][0]
    reloaded = image_processor.preprocess(recovered_image, do_normalize=True, return_tensors="pt")
    reloaded_pixels = reloaded["pixel_values"].float().to(device)
    reloaded_sizes = reloaded["image_sizes"].to(device)
    with torch.no_grad():
        recovered_tokens = base_model.get_image_tokens(reloaded_pixels, reloaded_sizes).to(device)
        match = int((gt_tokens == recovered_tokens).sum().item())
        print(f"[{tag}] Final verification (post -> pre round-trip): {match}/{len(gt_tokens)} tokens match")
    recovered_image.save(f"output_images/recovered_{tag}.png")

# ---------------- Setup 1 ----------------
# Reload noisy pixel values from the saved PNG; optimize toward ground_truth_tokens.
print("\n===== Setup 1: pixels from PNG reload =====")
reloaded_image_1 = Image.open("output_images/generated.png")
pre_1 = image_processor.preprocess(reloaded_image_1, do_normalize=True, return_tensors="pt")
noisy_pixel_values_1 = pre_1["pixel_values"].float().to(device)
vq_indices_1 = bpe_to_vq_indices(ground_truth_tokens, height, width)

# ---------------- Setup 2 ----------------
# Encode the PNG back to tokens, decode those tokens to pixels, and optimize those
# pixels back toward the tokens they came from — a self-consistent target.
print("\n===== Setup 2: tokens encoded from PNG, then decoded =====")
reloaded_image_2 = Image.open("output_images/generated.png")
pre_2 = image_processor.preprocess(reloaded_image_2, do_normalize=True, return_tensors="pt")
reloaded_pixels_2 = pre_2["pixel_values"].float().to(device)
reloaded_sizes_2 = pre_2["image_sizes"].to(device)
with torch.no_grad():
    # get_image_tokens returns BPE-with-EOL tokens (same layout as ground_truth_tokens)
    bpe_tokens_2 = base_model.get_image_tokens(reloaded_pixels_2, reloaded_sizes_2).to(device).flatten()
    vq_indices_2 = bpe_to_vq_indices(bpe_tokens_2, height, width)
    # decode_image_tokens also expects BPE-with-EOL + trailing specials
    padded_tokens_2 = torch.cat(
        [bpe_tokens_2,
         torch.zeros(3, dtype=bpe_tokens_2.dtype, device=device)])
    noisy_pixel_values_2 = base_model.decode_image_tokens(
        padded_tokens_2.unsqueeze(0),
        height=(height // downsample_ratio),
        width=(width // downsample_ratio))
torch.cuda.empty_cache()

# ---------------- Diagnostic pre-pass ----------------
# Before any optimization, measure how far each setup's encoder output sits from
# its target codebook embeddings. Setup 2's targets should be self-consistent
# (near-zero distance); Setup 1's targets came from a different decode pass.
print("\n===== Diagnostic pre-pass =====")
diag_1 = diagnose(noisy_pixel_values_1, vq_indices_1, tag="setup1")
diag_2 = diagnose(noisy_pixel_values_2, vq_indices_2, tag="setup2")
plot_diagnostics([diag_1, diag_2])

# ---------------- Optimization ----------------
recovered_1 = optimize_pixels(noisy_pixel_values_1, vq_indices_1, height, width, n_iterations=1000, patience=100)
verify_and_save(recovered_1, noisy_pixel_values_1, ground_truth_tokens, tag="setup1")

# recovered_2 = optimize_pixels(noisy_pixel_values_2, vq_indices_2, height, width, n_iterations=1000)
# verify_and_save(recovered_2, noisy_pixel_values_2, bpe_tokens_2, tag="setup2")