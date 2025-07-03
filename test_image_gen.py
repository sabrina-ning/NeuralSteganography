from PIL import Image
import torch
from transformers import AutoProcessor, Emu3ForConditionalGeneration
import numpy as np
import time
import ipdb

# setup
ipdb.set_trace()
start = time.time()
np.set_printoptions(threshold=np.inf)

# set deterministic seed
seed = 12345
torch.random.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

model_id = "BAAI/Emu3-Gen-hf"
model = Emu3ForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True,
    device_map="cuda:0",
)
base_model = model.model
processor = AutoProcessor.from_pretrained(model_id)

# preprocess prompt for image generation
# conversation = [
#     {"role": "user",
#      "content": [{"type": "text",
#                   "text": "Generate an image of a happy dog."}]},
# ]
# prompt = processor.apply_chat_template(
#     conversation, 
#     add_generation_prompt=True, 
#     tokenize=False
# )
inputs = processor(
    # text=[prompt],
    text = ["a puppy"],
    padding=True,
    return_tensors="pt",
    return_for_image_generation=True,
).to(model.device)

print(inputs)
image_sizes = inputs.pop("image_sizes") # intended output image dimensions
HEIGHT, WIDTH = image_sizes[0] # (90, 90)
VISUAL_TOKENS = base_model.vocabulary_mapping.image_tokens # [151854, 184621]

def prefix_allowed_tokens_fn(batch_id, input_ids):
    # get special token ids
    image_wrapper_token_id = torch.tensor([processor.tokenizer.image_wrapper_token_id], device=model.device)
    eoi_token_id = torch.tensor([processor.tokenizer.eoi_token_id], device=model.device)
    eos_token_id = torch.tensor([processor.tokenizer.eos_token_id], device=model.device)
    pad_token_id = torch.tensor([processor.tokenizer.pad_token_id], device=model.device)
    eof_token_id = torch.tensor([processor.tokenizer.eof_token_id], device=model.device)
    eol_token_id = processor.tokenizer.encode("<|extra_200|>", return_tensors="pt")[0]

    # calculate offset (number of tokens generated for image so far)
    position = torch.nonzero(input_ids == image_wrapper_token_id, as_tuple=True)[0][0]
    offset = input_ids.shape[0] - position
    if offset % (WIDTH + 1) == 0: # finished a row of visual tokens, next must be 'end of line'
        return (eol_token_id,)
    elif offset == (WIDTH + 1) * HEIGHT + 1: # finished generating all image tokens, next must be 'end of frame'
        return (eof_token_id,)
    elif offset == (WIDTH + 1) * HEIGHT + 2:
        return (eoi_token_id,)
    elif offset == (WIDTH + 1) * HEIGHT + 3:
        return (eos_token_id,)
    elif offset > (WIDTH + 1) * HEIGHT + 3: # if forced to continue generating tokens
        return (pad_token_id,)
    else:
        return VISUAL_TOKENS

# breakpoint()
out = model.generate(
    **inputs,
    max_new_tokens=9000,
    prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
    return_dict_in_generate=True,
    do_sample=True, # samples from distribution of likely next tokens, instead of greedy
    num_beams=1,
    use_cache=True,
    temperature=1.0,
    top_k=2048
)
# breakpoint()

image_tokens = out.sequences[:, inputs.input_ids.shape[1]: ] # removes original prompt tokens
print("="*40)
# print(out.sequences.cpu().numpy())
print(image_tokens.cpu().numpy()) # discrete tokens
print("="*40)
image = base_model.decode_image_tokens(image_tokens.cuda(), height=HEIGHT, width=WIDTH)

# pixel_values = processor.postprocess(image, return_tensors="pt")['pixel_values'].unsqueeze(0).unsqueeze(0) # (B, T, C, H, W)
# image_tokens = base_model.get_image_features(pixel_values, [[720, 720]])[0]
# print(image_tokens.cpu().numpy())

# print(image_tensor['pixel_values'])
# torch.save(image_tensor['pixel_values'], "result.pt")

image = processor.image_processor.postprocess(image, return_tensors="PIL.Image.Image")
image = image['pixel_values'][0]
image.save("result_new.png")

end = time.time()
print(f"Took {end - start:.2f} sec")