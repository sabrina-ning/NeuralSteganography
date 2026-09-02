"""Pixel-level optimization to recover target VQ tokens from an image.

Given a desired VQ token sequence (e.g., from arithmetic coding), this module
optimizes the pixel values of a decoded image so that re-encoding through the
VQ encoder produces exactly the target tokens.

Key techniques:
- Straight-through estimator (STE) to simulate PNG uint8 quantization during optimization
- Custom spatial-only encoder (bypasses redundant temporal frames for still images)
- Focused loss on mismatched tokens when most tokens already match
- Random restarts with targeted noise on mismatched token regions
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


def optimize_pixels(noisy_pixel_values, original_tokens, height, width,
                    base_model, image_processor,
                    n_iterations=2000, lr=1e-3, patience=200, use_ste=True):
    """Optimize decoded pixel values so they round-trip through PNG save/load
    back to the original BPE image tokens.

    Args:
        noisy_pixel_values: decoded pixel tensor [1, 3, H, W] (from decode_image_tokens)
        original_tokens: 1-D tensor of BPE image tokens (with EOL markers, no end tokens)
        height: image height in pixels
        width: image width in pixels
        base_model: the Emu3Model (e.g. model.model for Emu3ForConditionalGeneration)
        image_processor: the processor's image_processor
        n_iterations: max optimization iterations
        lr: learning rate
        patience: iterations without improvement before random restart
        use_ste: whether to apply straight-through estimator for PNG quantization
    Returns:
        optimized pixel tensor [1, 3, H, W] in float32
    """
    device = noisy_pixel_values.device
    downsample_ratio = image_processor.spatial_factor
    vqmodel = base_model.vqmodel
    original_dtype = next(vqmodel.parameters()).dtype
    vqmodel.float()  # float32 needed for optimization gradients
    temporal = vqmodel.config.temporal_downsample_factor

    # Get ground truth codebook embeddings from original tokens
    bpe_grid = original_tokens.view(height // downsample_ratio, width // downsample_ratio + 1)
    vq_indices = base_model.vocabulary_mapping.convert_bpe2img(bpe_grid).flatten().to(device)
    target_embeddings = vqmodel.quantize.embedding(vq_indices).detach()

    modified = noisy_pixel_values.clone().float().requires_grad_(True)
    optimizer = torch.optim.Adam([modified], lr=lr)
    losses = []
    best_matches = 0
    best_pixels = modified.detach().clone()
    iters_since_improvement = 0

    enc = vqmodel.encoder

    def _quantize_ste(pv):
        """Simulate PNG uint8 round-trip with straight-through estimator.

        Uses float64 precision to match NumPy's behavior in postprocess,
        then applies STE so gradients flow through the quantization.
        """
        pv_255 = (pv.double() * 0.5 + 0.5) * 255.0
        pv_floored = pv_255.floor().clamp(0, 255)
        pv_ste = pv_255 + (pv_floored - pv_255).detach()
        pv_rescaled = pv_ste * image_processor.rescale_factor
        return ((pv_rescaled - 0.5) / 0.5).float()

    def _spatial_encode(pv):
        """Run spatial encoder on a single frame.

        Emu3's VQ encoder normally processes 4 identical temporal frames for
        still images. This function bypasses the redundant temporal expansion
        at the spatial stage, saving ~4x memory.
        """
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

    def _full_encode(pv):
        """Run the full encoder (spatial + temporal + quant_conv) on pixel values."""
        h = _spatial_encode(pv)
        h = h.unsqueeze(2).expand(-1, -1, temporal, -1, -1)
        for t_conv in enc.time_conv:
            h = t_conv(h)
            h = h * torch.sigmoid(h)
        for t_res in enc.time_res_stack:
            h = t_res(h)
        h = h.permute(0, 2, 1, 3, 4)
        c = h.permute(0, 2, 1, 3, 4)
        c = vqmodel.quant_conv(c)
        c = c.permute(0, 2, 1, 3, 4)
        return c

    _maybe_quantize = _quantize_ste if use_ste else (lambda pv: pv)

    # Diagnostic: how many tokens does the initial decoded image already recover?
    with torch.no_grad():
        initial_conv = _full_encode(noisy_pixel_values.float())
        initial_tokens = vqmodel.quantize(initial_conv).flatten()
        initial_matches = int((initial_tokens == vq_indices).sum().item())
        print(f"DIAGNOSTIC: initial round-trip (no optim, float32): "
              f"{initial_matches}/{len(vq_indices)}")

    for i in range(n_iterations):
        optimizer.zero_grad()

        conv = _full_encode(_maybe_quantize(modified))
        conv_flat = conv.squeeze(0).squeeze(0).permute(1, 2, 0).reshape(-1, 4)

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
            iters_since_improvement = 0
        else:
            iters_since_improvement += 1

        # Random restart: perturb from best state with targeted noise
        if iters_since_improvement >= patience:
            noise_scale = 0.01
            print(f"iter {i}: RESTART -- stuck at {matches}/{len(vq_indices)} for {patience} iters "
                  f"(best={best_matches}), adding noise (scale={noise_scale})")
            with torch.no_grad():
                mismatched = current_indices != vq_indices
                h_tokens = height // downsample_ratio
                w_tokens = width // downsample_ratio
                mis_2d = mismatched.view(h_tokens, w_tokens).float()
                mis_mask = F.interpolate(
                    mis_2d.unsqueeze(0).unsqueeze(0),
                    size=(height, width), mode='nearest')
                noise = torch.randn_like(best_pixels) * noise_scale
                noise[mis_mask.expand_as(noise) < 0.5] = 0
                modified.copy_(best_pixels + noise)
            # Reset Adam state — stale momentum would catapult pixels away
            optimizer = torch.optim.Adam([modified], lr=lr)
            del conv, conv_flat, current_indices
            torch.cuda.empty_cache()
            iters_since_improvement = 0
            continue

        loss = F.mse_loss(conv_flat, target_embeddings)
        loss.backward()

        optimizer.step()
        losses.append(loss.item())

        if i % 10 == 0:
            with torch.no_grad():
                mismatched = current_indices != vq_indices
                if mismatched.any():
                    mis_flat = conv_flat[mismatched]
                    mis_target = target_embeddings[mismatched]
                    mis_current = vqmodel.quantize.embedding(current_indices[mismatched])
                    dist_to_target = (mis_flat - mis_target).pow(2).sum(1)
                    dist_to_current = (mis_flat - mis_current).pow(2).sum(1)
                    gap = dist_to_target - dist_to_current
                    print(f"iter {i}: loss {loss.item():.6f}, tokens matching: {matches}/{len(vq_indices)} | "
                          f"mismatched gap (mean={gap.mean().item():.6f}, max={gap.max().item():.6f})")
                else:
                    print(f"iter {i}: loss {loss.item():.6f}, tokens matching: {matches}/{len(vq_indices)}")

    # Plot loss curve
    plt.figure()
    plt.plot(losses)
    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.title("optimization loss curve")
    plt.savefig("output_images/loss_curve.png")
    plt.close()
    print(f"optimization done: {best_matches}/{len(vq_indices)} tokens matched")

    # Restore vqmodel to original dtype
    vqmodel.to(original_dtype)

    return best_pixels
