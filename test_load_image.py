from PIL import Image
from transformers import AutoProcessor, Emu3Model
import torch
import torch.nn.functional as F
import gc

# Global constants
model_id = "BAAI/Emu3-Gen-hf"
image_path1 = "test_images/dog_720x720.png"
image_path2 = "robustness_results/recovered_dog_720x720_seed1.png"
device = "cuda" if torch.cuda.is_available() else "cpu"

print("original:", image_path1)
print("recovered:", image_path2)

print(f"Loading model {model_id}")
base_model = Emu3Model.from_pretrained(model_id, dtype=torch.bfloat16)
vqmodel = base_model.vqmodel.float()  # float32 to match optimizer precision
vqmodel.to(device)
base_model.eval()

processor = AutoProcessor.from_pretrained(model_id, dtype=torch.bfloat16)
image_processor = processor.image_processor
image_processor.do_resize = False

def get_tokens_simple(path):
    image = Image.open(path).convert("RGB")
    inputs = image_processor.preprocess(image, do_normalize=True, return_tensors="pt")
    pixel_values = inputs["pixel_values"].float().to(device)
    image_sizes = inputs["image_sizes"].to(device)
    with torch.no_grad():
        tokens = base_model.get_image_tokens(pixel_values, image_sizes)
    return tokens

original_tokens = get_tokens_simple(image_path1)
recovered_tokens = get_tokens_simple(image_path2)

match_count = (original_tokens == recovered_tokens).sum().item()
total_tokens = len(original_tokens)

print(f"Match count: {match_count} / {total_tokens}")
if match_count == total_tokens:
    print("SUCCESS: All tokens match!")
    breakpoint()
else:
    print(f"FAILURE: {total_tokens - match_count} tokens mismatch.")
    diff = (original_tokens != recovered_tokens)
    mismatched_indices = torch.where(diff)[0]
    print(f"First 10 mismatched indices: {mismatched_indices[:10].tolist()}")
