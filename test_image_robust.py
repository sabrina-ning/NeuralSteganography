from PIL import Image
from transformers import AutoProcessor, Emu3Model
import ipdb
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Global constants
model_id = "BAAI/Emu3-Gen-hf"
image_path = "test_images/cat_720x720.png"
torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(image_path)

base_model = Emu3Model.from_pretrained(
  model_id,
  dtype=torch.bfloat16,
  device_map=device)
vqmodel = base_model.vqmodel.float()  # float32 for precision

processor = AutoProcessor.from_pretrained(
  model_id,
  dtype=torch.bfloat16,
  device_map=device)
image_processor = processor.image_processor
image_processor.do_resize = False
downsample_ratio = image_processor.spatial_factor

image = Image.open(image_path)
image = image_processor.preprocess(image, do_normalize=True, return_tensors="pt")
pixel_values = image["pixel_values"].float().to(device)
image_sizes = image["image_sizes"].to(device)
height, width = image_sizes[0]
print("image sizes:", image_sizes)

with torch.no_grad():
    ground_truth_tokens = base_model.get_image_tokens(pixel_values, image_sizes).to(device)
    print("ground truth tokens:", ground_truth_tokens)

padded_tokens = torch.cat([ground_truth_tokens, torch.zeros(3, dtype=ground_truth_tokens.dtype, device=device)])
with torch.no_grad():
    decoded_pixels = base_model.decode_image_tokens(
        padded_tokens.unsqueeze(0),
        height=(height // downsample_ratio),
        width=(width // downsample_ratio))
torch.cuda.empty_cache()

def optimize_pixels(noisy_pixel_values, original_tokens, height, width,
                    n_iterations=2000, lr=1e-3, patience=200):
    # original_tokens is a tensor of BPE tokens with EOL markers

    temporal = vqmodel.config.temporal_downsample_factor

    # Get ground truth codebook embeddings from original tokens
    bpe_grid = original_tokens.view(height // downsample_ratio, width // downsample_ratio + 1) # reshape to grid
    vq_indices = base_model.vocabulary_mapping.convert_bpe2img(bpe_grid).flatten().to(device) # strips EOL markers internally
    target_embeddings = vqmodel.quantize.embedding(vq_indices).detach()

    modified = noisy_pixel_values.clone().float().requires_grad_(True)
    optimizer = torch.optim.Adam([modified], lr=lr)
    losses = []
    prev_matches = 0
    best_matches = 0
    best_pixels = modified.detach().clone()
    iters_since_improvement = 0

    enc = vqmodel.encoder

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

    for i in range(n_iterations):
        optimizer.zero_grad()

        hidden = _spatial_encode(_quantize_ste(modified))  # [1, 4, H', W']

        # Expand to temporal dimension, run lightweight temporal convs
        hidden = hidden.unsqueeze(2).expand(-1, -1, temporal, -1, -1)  # [1, 4, temporal, H', W']
        for t_conv in enc.time_conv:
            hidden = t_conv(hidden)
            hidden = hidden * torch.sigmoid(hidden)
        for t_res in enc.time_res_stack:
            hidden = t_res(hidden)
        hidden = hidden.permute(0, 2, 1, 3, 4) # [1, temporal=1, lat_ch, H', W']

        conv = hidden.permute(0, 2, 1, 3, 4) # [1, lat_ch, 1, H', W']
        conv = vqmodel.quant_conv(conv)
        conv = conv.permute(0, 2, 1, 3, 4) # [1, 1, lat_ch, H', W']

        # Reshape encoder output into [n_tokens, 4] to match codebook embeddings
        conv_flat = conv.squeeze(0).squeeze(0).permute(1, 2, 0).reshape(-1, 4)

        # Check how many current tokens match the original ground truth
        with torch.no_grad():
            current_indices = vqmodel.quantize(conv.detach()).flatten()
            matches = int((current_indices == vq_indices).sum().item())

        if matches == len(vq_indices):
            print(f"iter {i}: all tokens matched")
            best_pixels = modified.detach().clone()
            best_matches = matches
            break

        # Track best and check for stall
        if matches > best_matches:
            best_matches = matches
            best_pixels = modified.detach().clone()
            iters_since_improvement = 0
        else:
            iters_since_improvement += 1

        # TODO: check if this is ever used
        # Random restart: perturb from best state with small noise
        if iters_since_improvement >= patience:
            noise_scale = 0.01
            print(f"iter {i}: RESTART — stuck at {matches}/{len(vq_indices)} for {patience} iters "
                  f"(best={best_matches}), adding noise (scale={noise_scale})")
            with torch.no_grad():
                # Only perturb pixels corresponding to mismatched tokens
                mismatched = current_indices != vq_indices
                h_tokens = height // downsample_ratio
                w_tokens = width // downsample_ratio
                mis_2d = mismatched.view(h_tokens, w_tokens).float()
                mis_mask = F.interpolate(
                    mis_2d.unsqueeze(0).unsqueeze(0),
                    size=(height, width), mode='nearest')  # [1, 1, H, W]
                noise = torch.randn_like(best_pixels) * noise_scale
                noise[mis_mask.expand_as(noise) < 0.5] = 0 # mis_mask is 0 for matched, 1 for mismatched
                modified.copy_(best_pixels + noise)
            del hidden, conv, conv_flat, current_indices
            torch.cuda.empty_cache()
            iters_since_improvement = 0
            prev_matches = 0
            continue

        mismatched = current_indices != vq_indices
        match_ratio = matches / len(vq_indices)
        use_focused = match_ratio >= 0.99 and matches >= prev_matches
        # use_focused = True
        if use_focused:
            scale = mismatched.sum().item() / len(vq_indices)
            loss = F.mse_loss(conv_flat[mismatched], target_embeddings[mismatched]) * scale
        else:
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
    plt.savefig("loss_curve.png")
    plt.close()
    print("loss curve saved to loss_curve.png")

    return best_pixels # float32, matches vqmodel precision

recovered_pixel_values = optimize_pixels(decoded_pixels, ground_truth_tokens, height, width)
delta = (recovered_pixel_values - decoded_pixels).abs()
print(f"Pixel range: [{decoded_pixels.min().item():.4f}, {decoded_pixels.max().item():.4f}]")
print(f"Max pixel change: {delta.max().item():.6f}")
print(f"Min pixel change: {delta.min().item():.6f}")
print(f"Mean pixel change: {delta.mean().item():.6f}")

# Verify via actual postprocess→preprocess round-trip
recovered_image = image_processor.postprocess(
    recovered_pixel_values.float(), return_tensors="PIL.Image.Image")["pixel_values"][0]
reloaded = image_processor.preprocess(recovered_image, do_normalize=True, return_tensors="pt")
reloaded_pixels = reloaded["pixel_values"].float().to(device)
reloaded_sizes = reloaded["image_sizes"].to(device)
with torch.no_grad():
    recovered_tokens = base_model.get_image_tokens(reloaded_pixels, reloaded_sizes).to(device)
    match = int((ground_truth_tokens == recovered_tokens).sum().item())
    print(f"Final verification (post -> pre round-trip): {match}/{len(ground_truth_tokens)} tokens match")

recovered_image.save("output_images/recovered.png")