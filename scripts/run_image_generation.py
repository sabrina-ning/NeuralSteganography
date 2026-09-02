from PIL import Image
import torch
import torch.nn.functional as F
from transformers import AutoProcessor, Emu3ForConditionalGeneration
import numpy as np
import time

## ===== SETUP =====

def setup(prompt_str="a kitten", seed=12345, model_id="BAAI/Emu3-Gen-hf"):
    model = Emu3ForConditionalGeneration.from_pretrained(
        model_id,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="cuda:0",
        # attn_implementation="flash_attention_2"
    )
    processor = AutoProcessor.from_pretrained(model_id)
    print("Model:", type(model))
    print("Processor:", type(processor))
    print("Tokenizer:", type(processor.tokenizer))

    inputs = processor(
        text=[prompt_str],
        padding=True,
        return_tensors="pt",
        return_for_image_generation=True,
    ).to(model.device)

    return model, processor, inputs, seed

'''
Example:

{'input_ids': tensor([[151849, 64, 41189, 151852, 24, 15, 9, 24, 15, 151851]], device='cuda:0'),
    'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], device='cuda:0'),
    'image_sizes': tensor([[90, 90]], device='cuda:0')}

<|extra_203|>aĠpuppy<|image start|>90*90<|image token|>

image_wrapper_token_id = 151851 = <|image token|>
'''

## ===== GENERATION FUNCTIONS =====

def model_generate(model, processor, inputs, seed):
    """Generate image tokens via HuggingFace `generate` with a prefix-constrained token fn."""
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    device = model.device
    base_model = model.model
    visual_tokens = base_model.vocabulary_mapping.image_tokens
    num_prompt_tokens = inputs["input_ids"].shape[1]
    height, width = inputs["image_sizes"][0]

    image_wrapper_token_id = processor.tokenizer.image_wrapper_token_id
    eol_token_id = processor.tokenizer.encode("<|extra_200|>", return_tensors="pt")[0].to(device)
    eof_token_id = torch.tensor([processor.tokenizer.eof_token_id], device=device)
    eoi_token_id = torch.tensor([processor.tokenizer.eoi_token_id], device=device)
    eos_token_id = torch.tensor([processor.tokenizer.eos_token_id], device=device)
    pad_token_id = torch.tensor([processor.tokenizer.pad_token_id], device=device)

    def prefix_allowed_tokens_fn(_, input_ids):
        position = torch.nonzero(input_ids == image_wrapper_token_id, as_tuple=True)[0][0]
        offset = input_ids.shape[0] - position

        if offset % (width + 1) == 0:
            return (eol_token_id,)
        elif offset == (width + 1) * height + 1:
            return (eof_token_id,)
        elif offset == (width + 1) * height + 2:
            return (eoi_token_id,)
        elif offset == (width + 1) * height + 3:
            return (eos_token_id,)
        elif offset > (width + 1) * height + 3:
            return (pad_token_id,)
        else:
            return visual_tokens

    out = model.generate(
        **inputs,
        max_new_tokens=50000,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        return_dict_in_generate=True,
        do_sample=True,
        num_beams=1,
        temperature=1.0,
        top_k=2048,
        use_cache=True,
    )
    return out.sequences[:, num_prompt_tokens:]

## ====================

def manual_generate(model, processor, inputs, seed, temp=1.0, top_k=2048):
    """Autoregressively generate image tokens by manually masking invalid tokens at each step."""
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    device = model.device
    base_model = model.model
    visual_tokens = base_model.vocabulary_mapping.image_tokens
    num_prompt_tokens = inputs["input_ids"].shape[1]
    height, width = inputs["image_sizes"][0]
    total_image_tokens = height * (width + 1)
    num_tokens = total_image_tokens + 3  # 3 extra special tokens at end

    image_wrapper_token_id = torch.tensor([processor.tokenizer.image_wrapper_token_id], device=device)
    eol_token_id = processor.tokenizer.encode("<|extra_200|>", return_tensors="pt")[0].to(device)
    eof_token_id = torch.tensor([processor.tokenizer.eof_token_id], device=device)
    eoi_token_id = torch.tensor([processor.tokenizer.eoi_token_id], device=device)
    eos_token_id = torch.tensor([processor.tokenizer.eos_token_id], device=device)

    context = inputs["input_ids"][0]
    prev = context
    output = context
    past = None

    with torch.no_grad():
        for i in range(num_tokens):
            out = model(prev.unsqueeze(0), past_key_values=past, use_cache=True)
            logits = out.logits[0, -1, :].to(dtype=torch.float32)
            past = out.past_key_values

            position = torch.nonzero(output == image_wrapper_token_id, as_tuple=True)[0][0]
            offset = output.shape[0] - position

            # prefix-constrained mask
            mask = torch.full_like(logits, -float('inf'), device=device)
            if offset % (width + 1) == 0:
                mask[eol_token_id] = 0.0
            elif offset == (width + 1) * height + 1:
                mask[eof_token_id] = 0.0
            elif offset == (width + 1) * height + 2:
                mask[eoi_token_id] = 0.0
            elif offset == (width + 1) * height + 3:
                mask[eos_token_id] = 0.0
            else:
                mask[visual_tokens[0]:] = 0.0
            logits = logits + mask

            # temperature + top-k
            logits = logits / temp
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits = logits.masked_fill(indices_to_remove, -float('inf'))

            probs = F.softmax(logits, dim=-1)
            prev = torch.multinomial(probs, num_samples=1)
            output = torch.cat((output, prev), dim=0)

            if i % 100 == 0:
                print(f"Generated {i+1}/{num_tokens} tokens")

    return output[num_prompt_tokens:].unsqueeze(0)

## ====================

# directly takes in image as visual tokens
def reconstruct(model, processor, inputs, seed, image_tokens, rows_to_predict=8, temp=1.0, top_k=2048):
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    device = model.device
    base_model = model.model
    visual_tokens = base_model.vocabulary_mapping.image_tokens
    num_prompt_tokens = inputs["input_ids"].shape[1]
    height, width = inputs["image_sizes"][0]

    image_wrapper_token_id = torch.tensor([processor.tokenizer.image_wrapper_token_id], device=device)
    eol_token_id = processor.tokenizer.encode("<|extra_200|>", return_tensors="pt")[0].to(device)
    eof_token_id = torch.tensor([processor.tokenizer.eof_token_id], device=device)
    eoi_token_id = torch.tensor([processor.tokenizer.eoi_token_id], device=device)
    eos_token_id = torch.tensor([processor.tokenizer.eos_token_id], device=device)

    prompt = inputs["input_ids"][0]
    context = torch.cat([prompt, image_tokens[:(height - rows_to_predict) * (width + 1)]])
    num_tokens = rows_to_predict * (width + 1) + 3

    prev = context
    output = context
    past = None

    with torch.no_grad():
        for i in range(num_tokens):
            out = model(prev.unsqueeze(0), past_key_values=past, use_cache=True)
            logits = out.logits[0, -1, :].to(dtype=torch.float32)
            past = out.past_key_values

            position = torch.nonzero(output == image_wrapper_token_id, as_tuple=True)[0][0]
            offset = output.shape[0] - position

            mask = torch.full_like(logits, -float('inf'), device=device)
            if offset % (width + 1) == 0:
                mask[eol_token_id] = 0.0
            elif offset == (width + 1) * height + 1:
                mask[eof_token_id] = 0.0
            elif offset == (width + 1) * height + 2:
                mask[eoi_token_id] = 0.0
            elif offset == (width + 1) * height + 3:
                mask[eos_token_id] = 0.0
            else:
                mask[visual_tokens[0]:] = 0.0
            logits = logits + mask

            logits = logits / temp
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits = logits.masked_fill(indices_to_remove, -float('inf'))

            probs = F.softmax(logits, dim=-1)
            prev = torch.multinomial(probs, num_samples=1)
            output = torch.cat((output, prev), dim=0)

            if i % 100 == 0:
                print(f"Generated {i+1}/{num_tokens} tokens")

    return output[num_prompt_tokens:].unsqueeze(0)

## ====================

def test_generate_and_reconstruct(model, processor, inputs, seed):
    base_model = model.model
    height, width = inputs["image_sizes"][0]

    print("="*40 + " image tokens new " + "="*40)
    image_tokens_new = manual_generate(model, processor, inputs, seed)
    print(image_tokens_new[0][0:20])
    print(image_tokens_new[0][-20:])
    print(image_tokens_new.shape)
    print("="*40)

    torch.save(image_tokens_new, "image_tensor_new.pt")

    # Decode visual tokens
    image_pixels = base_model.decode_image_tokens(image_tokens_new, height=height, width=width)
    image = processor.image_processor.postprocess(image_pixels, return_tensors="PIL.Image.Image")['pixel_values'][0]
    image.save("result_new.png")

    # Reconstructing
    image_path = "result_new.png"
    image = Image.open(image_path).convert("RGB")

    image = processor.image_processor.preprocess(image, return_tensors="pt")
    pixel_values = image["pixel_values"].to(torch.float16).cuda()
    image_sizes = image["image_sizes"].cuda()

    with torch.no_grad():
        image_tokens = base_model.get_image_tokens(pixel_values, image_sizes).cuda()

    print("="*40 + " rec image tokens " + "="*40)
    rec_image_tokens = reconstruct(model, processor, inputs, seed, image_tokens)
    print(rec_image_tokens[0][-20:])
    print(rec_image_tokens.shape)
    print("="*40)

    rec_image_pixels = base_model.decode_image_tokens(rec_image_tokens, height=height, width=width)
    rec_image = processor.image_processor.postprocess(rec_image_pixels, return_tensors="PIL.Image.Image")['pixel_values'][0]
    rec_image.save("result_new_recon.png")


if __name__ == "__main__":
    start = time.time()
    np.set_printoptions(threshold=np.inf)
    model, processor, inputs, seed = setup()
    test_generate_and_reconstruct(model, processor, inputs, seed)
    end = time.time()
    print(f"Took {end - start:.2f} sec")
