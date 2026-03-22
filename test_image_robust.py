from PIL import Image
from transformers import AutoProcessor, Emu3Model
import ipdb
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Global constants
model_id = "BAAI/Emu3-Gen-hf"
image_path = "test_images/nature_256x256.png"
# image_path = "test_images/dog_720x720.png"
torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model to CPU first, then move parts to GPU as needed
base_model = Emu3Model.from_pretrained(
  model_id,
  torch_dtype=torch.bfloat16)
base_model.to(device)
vqmodel = base_model.vqmodel.float()  # float32 for precision

# vqmodel.encoder = torch.compile(vqmodel.encoder)
# vqmodel.quant_conv = torch.compile(vqmodel.quant_conv)

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

print("original pixel values:", pixel_values)
print("image sizes:", image_sizes)

with torch.no_grad():
    ground_truth_tokens = base_model.get_image_tokens(pixel_values, image_sizes).to(device)
    print("ground truth tokens:", ground_truth_tokens)

# Decode tokens back to pixels (this will not re-encode to the same tokens)
padded_tokens = torch.cat([ground_truth_tokens, torch.zeros(3, dtype=ground_truth_tokens.dtype, device=device)])
with torch.no_grad():
    decoded_pixels = base_model.decode_image_tokens(
        padded_tokens.unsqueeze(0),
        height=(height // downsample_ratio),
        width=(width // downsample_ratio))

# TODO: add heuristic to randomly restart if stuck, after N iterations with no improvement
def optimize_pixels(noisy_pixel_values, original_tokens, height, width, n_iterations=2000, lr=1e-3):
    # original_tokens is a tensor of BPE tokens with EOL markers

    temporal = vqmodel.config.temporal_downsample_factor

    # Get ground truth codebook embeddings from original tokens
    bpe_grid = original_tokens.view(height // downsample_ratio, width // downsample_ratio + 1) # reshape to grid
    vq_indices = base_model.vocabulary_mapping.convert_bpe2img(bpe_grid).flatten().to(device) # strips EOL markers internally
    target_embeddings = vqmodel.quantize.embedding(vq_indices).detach()

    modified = noisy_pixel_values.clone().float().requires_grad_(True)
    optimizer = torch.optim.Adam([modified], lr=lr)
    losses = []

    for i in range(n_iterations):
        optimizer.zero_grad()

        # Forward pass in float32 (vqmodel is float32, matches get_image_tokens path)
        pv_5d = modified.unsqueeze(1).repeat(1, temporal, 1, 1, 1)
        hidden = vqmodel.encoder(pv_5d)
        conv = hidden.permute(0, 2, 1, 3, 4)
        conv = vqmodel.quant_conv(conv)
        conv = conv.permute(0, 2, 1, 3, 4)  # [1, 1, 4, 32, 32]

        # Reshape encoder output into [n_tokens, 4] to match codebook embeddings
        conv_flat = conv.squeeze(0).squeeze(0).permute(1, 2, 0).reshape(-1, 4)

        # Check how many current tokens match the original ground truth
        # Use the actual quantizer to avoid floating-point discrepancies
        with torch.no_grad():
            print("here")
            current_indices = vqmodel.quantize(conv.detach()).flatten()
            matches = int((current_indices == vq_indices).sum().item())
            # Old manual distance computation (can disagree with quantizer on boundary tokens):
            # emb = vqmodel.quantize.embedding.weight
            # flat = conv_flat.detach()
            # current_indices = torch.cat([
            #     (chunk @ emb.T).mul(-2).add_(
            #         (chunk ** 2).sum(1, keepdim=True)).add_(
            #         (emb ** 2).sum(1)).argmin(1)
            #     for chunk in flat.split(512)
            # ])

        if matches == len(vq_indices):
            print(f"Successfully matched all tokens at iteration {i}")
            break

        loss = F.mse_loss(conv_flat, target_embeddings)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if i % 10 == 0:
            print(f"iter {i}: loss {loss.item():.6f}, tokens matching: {matches}/{len(vq_indices)}")
        
    # plotting
    plt.figure()
    plt.plot(losses)
    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.title("optimization loss curve")
    plt.savefig("loss_curve.png")
    plt.close()
    print("loss curve saved to loss_curve.png")

    return modified.detach() # float32, matches vqmodel precision


# Offload everything except encoder + quant_conv to CPU to free VRAM
base_model.text_model.to("cpu")
vqmodel.decoder.to("cpu")
vqmodel.post_quant_conv.to("cpu")
torch.cuda.empty_cache()

recovered_pixel_values = optimize_pixels(decoded_pixels, ground_truth_tokens, height, width)

# Bring everything back to GPU for verification
vqmodel.decoder.to(device)
vqmodel.post_quant_conv.to(device)
base_model.text_model.to(device)

# Verify recovered tokens match
with torch.no_grad():
    recovered_tokens = base_model.get_image_tokens(recovered_pixel_values, image_sizes).to(device)
    match = int((ground_truth_tokens == recovered_tokens).sum().item())
    print(f"Final verification: {match}/{len(ground_truth_tokens)} tokens match")

# Save recovered image
recovered_image = image_processor.postprocess(
    recovered_pixel_values.float(), return_tensors="PIL.Image.Image")['pixel_values'][0]
recovered_image.save("output_images/recovered.png")

breakpoint()