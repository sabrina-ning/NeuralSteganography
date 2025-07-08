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

prompt_str = "a puppy"
inputs = processor(
    text=[prompt_str],
    padding=True,
    return_tensors="pt",
    return_for_image_generation=True,
).to(model.device)

# print(inputs)
NUM_PROMPT_TOKENS = inputs["input_ids"].shape[1]
# print(NUM_PROMPT_TOKENS)
HEIGHT, WIDTH = inputs["image_sizes"][0] # output image dimensions (90, 90)
VISUAL_TOKENS = base_model.vocabulary_mapping.image_tokens # [151854, 184621]
VISUAL_TOKEN_START = 151854

# enforce valid image structure
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

# autoregressive token generation
def generate_tokens(model, context, num_tokens, temp=1.0, topk=2048):
    prev = context
    output = context
    past = None

    for i in range(num_tokens):
        eol_token_id = processor.tokenizer.encode("<|extra_200|>", return_tensors="pt")[0].cuda()

        offset = i + 1
        with torch.no_grad():
            out = model(prev.unsqueeze(0), past_key_values=past, use_cache=True)
            logits = out.logits
            past = out.past_key_values

            if offset % (WIDTH + 1) == 0: # finished a row of visual tokens, next must be 'end of line'
                prev = eol_token_id
            else: # next must be visual token
                logits = logits[0, -1, VISUAL_TOKEN_START:]
                logits_temp = logits / temp
                probs = F.softmax(logits_temp, dim=-1)
                topk_probs, topk_indices = torch.topk(probs, topk)
                if i < 100:
                    print(torch.multinomial(topk_probs, num_samples=1))
                prev = topk_indices[torch.multinomial(topk_probs, num_samples=1)]
                # breakpoint()
                prev += VISUAL_TOKEN_START

            output = torch.cat((output, prev), dim=0)

        if i % 100 == 0:
            print(i)

    print("output:", output.shape)
    print("generated rows:", output[-num_tokens:].shape)
    return output[NUM_PROMPT_TOKENS:]

def model_generate():
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

    return image

def manual_generate():
    with torch.no_grad():
        num_tokens = HEIGHT * (WIDTH + 1) # extra token on each line

        context = inputs["input_ids"][0]

        gen = generate_tokens(model, context, num_tokens)

        eof_token_id = processor.tokenizer.eof_token_id
        eoi_token_id = processor.tokenizer.eoi_token_id
        eos_token_id = processor.tokenizer.eos_token_id
        end_tokens = torch.tensor([eof_token_id, eoi_token_id, eos_token_id], device="cuda")
        
        gen = torch.cat([gen, end_tokens], dim=0)

        print(gen.cpu().numpy())
        # print(gen)
        image = base_model.decode_image_tokens(gen.unsqueeze(0).cuda(), height=HEIGHT, width=WIDTH)
        
    return image

# image = model_generate()
image = manual_generate()

image = processor.image_processor.postprocess(image, return_tensors="PIL.Image.Image")['pixel_values'][0]
image.save("result_new.png")

end = time.time()
print(f"Took {end - start:.2f} sec")