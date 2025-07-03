import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from transformers import AutoModel, AutoProcessor, Emu3ForConditionalGeneration, Emu3ImageProcessor
import ipdb

ipdb.set_trace()
np.set_printoptions(threshold=np.inf)

VISUAL_TOKEN_START = 151854

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

HEIGHT, WIDTH = 90, 90

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
                print(torch.multinomial(topk_probs, num_samples=1))
                prev = topk_indices[torch.multinomial(topk_probs, num_samples=1)]
                breakpoint()
                prev += VISUAL_TOKEN_START

            output = torch.cat((output, prev), dim=0)

        if i % 100 == 0:
            print(i)

    print("output:", output.shape)
    print("generated rows:", output[-num_tokens:].shape)
    return output[10:]

## ========== PREDICTING ==========

image_path = "result_new.png"
image = Image.open(image_path).convert("RGB") # returns Image object

# image = torch.load("result.pt")
# print(image.shape)

image = processor.image_processor.preprocess(image, return_tensors="pt") # returns BatchFeature object
pixel_values = image["pixel_values"].to(torch.float16).cuda()
image_sizes = image["image_sizes"].cuda()

with torch.no_grad():
    print("="*40 + " Encoding " + "="*40)
    image_tokens = base_model.get_image_tokens(pixel_values, image_sizes).cuda() # [8190]

    print("image tokens:", torch.min(image_tokens), torch.max(image_tokens))
    print(image_tokens.shape)

    eof_token_id = processor.tokenizer.eof_token_id
    eoi_token_id = processor.tokenizer.eoi_token_id
    eos_token_id = processor.tokenizer.eos_token_id

    ROWS_TO_PREDICT = 8
    TOKENS_PER_ROW = 91
    num_tokens = ROWS_TO_PREDICT * TOKENS_PER_ROW

    prompt = torch.tensor([151849, 64, 41189, 151852, 24, 15, 9, 24, 15, 151851], device="cuda")
    context = torch.cat([prompt, image_tokens[:-num_tokens]]) # (88*91)

    print("="*40 + " Predicting " + "="*40)

    recon = generate_tokens(model, context, num_tokens)
    end_tokens = torch.tensor([eof_token_id, eoi_token_id, eos_token_id], device="cuda")
    recon = torch.cat([recon, end_tokens], dim=0)

    print("reconstructed:", torch.min(recon), torch.max(recon))
    print(recon.cpu().numpy())
    
    print("="*40 + " Decoding " + "="*40)
    recon_image = base_model.decode_image_tokens(recon.unsqueeze(0).cuda(), height=HEIGHT, width=WIDTH)

recon_image = processor.postprocess(recon_image, return_tensors="PIL.Image.Image")["pixel_values"][0]
recon_image.save("result_new_recon2.png")
