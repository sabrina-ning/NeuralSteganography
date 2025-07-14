from PIL import Image
import torch
import torch.nn.functional as F
from transformers import AutoProcessor, Emu3ForConditionalGeneration
import numpy as np
import time
import ipdb

## ===== FOR DEBUGGING =====

# ipdb.set_trace()
start = time.time()
np.set_printoptions(threshold=np.inf)

## ===== SETUP =====

model_id = "BAAI/Emu3-Gen-hf"
model = Emu3ForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True,
    device_map="cuda:0",
    # attn_implementation="flash_attention_2"
)
base_model = model.model
processor = AutoProcessor.from_pretrained(model_id)
print(type(model), type(processor))

seed = 12345
prompt_str = "a puppy"
inputs = processor(
    text=[prompt_str],
    padding=True,
    return_tensors="pt",
    return_for_image_generation=True,
).to(model.device)

num_prompt_tokens = inputs["input_ids"].shape[1]
height, width = inputs["image_sizes"][0] # (90, 90)
total_image_tokens = height * (width + 1)

visual_tokens = base_model.vocabulary_mapping.image_tokens # [151854, 184621]

# special tokens
image_wrapper_token_id = torch.tensor([processor.tokenizer.image_wrapper_token_id], device=model.device)
eol_token_id = processor.tokenizer.encode("<|extra_200|>", return_tensors="pt")[0].to(device=model.device)
eof_token_id = torch.tensor([processor.tokenizer.eof_token_id], device=model.device)
eoi_token_id = torch.tensor([processor.tokenizer.eoi_token_id], device=model.device)
eos_token_id = torch.tensor([processor.tokenizer.eos_token_id], device=model.device)
pad_token_id = torch.tensor([processor.tokenizer.pad_token_id], device=model.device)

## ===== HELPER FUNCTIONS =====

# enforce valid image structure for transformers `generate`
def prefix_allowed_tokens_fn(_, input_ids):
    position = torch.nonzero(input_ids == image_wrapper_token_id, as_tuple=True)[0][0]
    offset = input_ids.shape[0] - position  # number of tokens generated so far

    if offset % (width + 1) == 0: # finished a row of visual tokens, next must be 'end of line'
        return (eol_token_id,)
    elif offset == (width + 1) * height + 1: # finished generating all image tokens, next must be 'end of frame'
        return (eof_token_id,)
    elif offset == (width + 1) * height + 2:
        return (eoi_token_id,)
    elif offset == (width + 1) * height + 3:
        return (eos_token_id,)
    elif offset > (width + 1) * height + 3: # if forced to continue generating tokens
        return (pad_token_id,)
    else:
        return visual_tokens

# autoregressive generation using masking to disallow invalid tokens
def generate_tokens(model, context, num_tokens, temp=1.0, top_k=2048):
    prev = context
    output = context
    past = None

    for i in range(num_tokens):
        with torch.no_grad():
            out = model(prev.unsqueeze(0), past_key_values=past, use_cache=True)
            logits = out.logits
            past = out.past_key_values

            position = torch.nonzero(output == image_wrapper_token_id, as_tuple=True)[0][0]
            offset = output.shape[0] - position

            logits = logits[0, -1, :].to(dtype=torch.float32)

            # PREFIX CONSTRAINED LOGITS PROCESSOR
            mask = torch.full_like(logits, -float('inf'), device=model.device)
            if offset % (width + 1) == 0: # finished a row of visual tokens, next must be 'end of line'
                mask[eol_token_id] = 0.0
            elif offset == (width + 1) * height + 1:
                mask[eof_token_id] = 0.0
            elif offset == (width + 1) * height + 2:
                mask[eoi_token_id] = 0.0
            elif offset == (width + 1) * height + 3:
                mask[eos_token_id] = 0.0
            else: # next must be visual token
                mask[visual_tokens[0]:] = 0.0
            masked_logits = logits + mask

            # TEMPERATURE LOGITS PROCESSOR
            logits_temp = masked_logits / temp # apply temp on masked logits

            # TOP-K LOGITS PROCESSOR
            indices_to_remove = logits_temp < torch.topk(logits_temp, top_k)[0][..., -1, None] # remove if prob less than last token of top-k
            next_token_logits = logits_temp.masked_fill(indices_to_remove, -float('inf'))

            probs = F.softmax(next_token_logits, dim=-1)
            prev = torch.multinomial(probs, num_samples=1)
            output = torch.cat((output, prev), dim=0)

        if i % 100 == 0:
            print(f"Generated {i+1}/{num_tokens} tokens")

    return output[num_prompt_tokens:]

## ====================

def model_generate():
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    out = model.generate(
        **inputs,
        max_new_tokens=50000,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        return_dict_in_generate=True,
        do_sample=True, # samples from distribution of likely next tokens, instead of greedy
        num_beams=1,
        temperature=1.0,
        top_k=2048,
        use_cache=True
    )
    image_tokens = out.sequences[:, num_prompt_tokens:] # removes original prompt tokens

    return image_tokens # [1, 8193]

## ====================

def manual_generate():
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    with torch.no_grad():
        context = inputs["input_ids"][0]
        image_tokens = generate_tokens(model, context, total_image_tokens + 3) # generate 3 extra special tokens at end

    return image_tokens.unsqueeze(0) # [1, 8193]

# directly takes in image as visual tokens
def reconstruct(image_tokens):
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    with torch.no_grad():
        rows_to_predict = 8
        num_tokens = rows_to_predict * (width + 1)

        prompt = inputs["input_ids"][0]
        context = torch.cat([prompt, image_tokens[:(height - rows_to_predict) * (width + 1)]])

        rec_image_tokens = generate_tokens(model, context, num_tokens)
    
    return rec_image_tokens.unsqueeze(0) # [1, 8193]

## ====================

print("="*40 + " image tokens " + "="*40)
image_tokens = model_generate()
print(image_tokens[0][0:20])
print(image_tokens[0][-20:])
print(image_tokens.shape)
print("="*40)

torch.save(image_tokens, "image_tensor.pt")

print("="*40 + " image tokens new " + "="*40)
image_tokens_new = manual_generate()
print(image_tokens_new[0][0:20])
print(image_tokens_new[0][-20:])
print(image_tokens_new.shape)
print("="*40)

torch.save(image_tokens_new, "image_tensor_new.pt")

# image_tokens = torch.load("image_tensor.pt")
# print(image_tokens.cpu().numpy())

# image_tokens_new = torch.load("image_tensor_new.pt")
# print(image_tokens_new.cpu().numpy())

print(image_tokens[0].tolist() == image_tokens_new[0].tolist())

# print("="*40 + " rec image tokens " + "="*40)
# rec_image_tokens = reconstruct(image_tokens[0].cuda())
# print(rec_image_tokens.cpu().numpy())
# print(rec_image_tokens.shape)
# print("="*40)

image_pixels = base_model.decode_image_tokens(image_tokens, height=height, width=width)
image = processor.image_processor.postprocess(image_pixels, return_tensors="PIL.Image.Image")['pixel_values'][0]
image.save("result.png")

image_pixels = base_model.decode_image_tokens(image_tokens_new, height=height, width=width)
image = processor.image_processor.postprocess(image_pixels, return_tensors="PIL.Image.Image")['pixel_values'][0]
image.save("result_new.png")

# rec_image_pixels = base_model.decode_image_tokens(rec_image_tokens, height=height, width=width)
# rec_image = processor.image_processor.postprocess(rec_image_pixels, return_tensors="PIL.Image.Image")['pixel_values'][0]
# rec_image.save("result_recon_new.png")

end = time.time()
print(f"Took {end - start:.2f} sec")
