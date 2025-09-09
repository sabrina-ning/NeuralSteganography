import torch
import numpy as np
import bitarray
from PIL import Image

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, Emu3ForConditionalGeneration, Emu3Processor, DynamicCache

def decode(self, token_ids, **kwargs):
    filtered_tokens = self.convert_ids_to_tokens(token_ids)
    text = self.convert_tokens_to_string(filtered_tokens)
    return text
AutoTokenizer.decode = decode

def _convert_token_to_id(self, token):
    return self.encoder.get(token, 0)
AutoTokenizer._convert_token_to_id = _convert_token_to_id


# handles both old and new cache formats
def limit_past(past):
    past = list(past)
    for i in range(len(past)):
        if isinstance(past[i], tuple):
            key, value = past[i]
            past[i] = (
                key[:, :, :, -1022:],
                value[:, :, :, -1022:]
            )
        else:
            past[i] = past[i][:, :, :, -1022:]
    return past

def kl(q, logq, logp):
    res = q*(logq-logp)/0.69315
    res[q==0] = 0
    return res.sum().item() # in bits

def entropy(q, logq):
    res = q*logq/0.69315
    res[q==0] = 0
    return -res.sum().item() # in bits

# e.g. [0, 1, 1, 1] looks like 1110=14
def bits2int(bits):
    res = 0
    for i, bit in enumerate(bits):
        res += bit*(2**i)
    return res

def int2bits(inp, num_bits):
    if num_bits == 0:
        return []
    strlist = ('{0:0%db}'%num_bits).format(inp)
    return [int(strval) for strval in reversed(strlist)]

def is_sent_finish(token_idx, enc):
    token = enc.tokenizer.decode([token_idx])
    return '.' in token or '!' in token or '?' in token

def num_same_from_beg(bits1, bits2):
    assert len(bits1) == len(bits2)
    for i in range(len(bits1)):
        if bits1[i] != bits2[i]:
            break
    return i

def encode_context(raw_text, enc):
    context_tokens = enc.tokenizer.encode('<|endoftext|>') + enc.tokenizer.encode(raw_text)
    return context_tokens

def encode_image(image_path, model, enc):
    image = Image.open(image_path).convert("RGB") # returns Image object
    # width, height = image.size
    image_PIL = image

    image = enc.image_processor.preprocess(image, return_tensors="pt")
    pixel_values = image["pixel_values"].to(torch.float16).cuda()
    # image_sizes = image["image_sizes"].cuda()
    image_sizes = torch.tensor([[64, 64]], device="cuda")
    height, width = image_sizes[0]

    with torch.no_grad():
        image_tokens = model.model.get_image_tokens(pixel_values, image_sizes).cuda() # pixel values -> tokens
    
    start_token = torch.tensor([enc.tokenizer.image_wrapper_token_id], device="cuda")
    end_tokens = torch.tensor([enc.tokenizer.eof_token_id, enc.tokenizer.eoi_token_id, enc.tokenizer.eos_token_id], device="cuda")
    # start_token = torch.tensor([151851], device="cuda")
    # end_tokens = torch.tensor([151847, 151853, 151850], device="cuda")
    image_tokens = torch.cat([start_token, image_tokens, end_tokens])

    return image_tokens, height, width, image_PIL

def decode_image(image_tokens, height, width, model, enc):
    if not isinstance(image_tokens, torch.Tensor):
        image_tokens = torch.tensor(image_tokens, device="cuda")
    image = model.model.decode_image_tokens(image_tokens.unsqueeze(0), # tokens -> pixel values
                                                height=(height // enc.image_processor.spatial_factor), 
                                                width=(width // enc.image_processor.spatial_factor))
    image = enc.image_processor.postprocess(image, return_tensors="PIL.Image.Image")['pixel_values'][0]
    return image

# Use gpt2-medium for 345M param model
# Use gpt2-large for 774M param model
def get_model(seed=1234, model_name='gpt2'):
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    enc = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map=device)
    enc.image_processor.min_pixels = 64 * 64
    
    if "hf" in model_name: # Emu3-Chat-hf or Emu3-Gen-hf
        model = Emu3ForConditionalGeneration.from_pretrained(
            model_name, 
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map=device)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map=device)
        
    model.eval()
    return enc, model # enc is processor

enc32_itoc = ['\0', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '.', ',', "'", '!', ' ']
enc32_ctoi = {k: v for v, k in enumerate(enc32_itoc)}
def enc32(text):
    bits = []
    for c in text:
        bits.extend(int2bits(enc32_ctoi[c], 5))
    return bits

def dec32(bits):
    text = ''
    for i in range(0, len(bits), 5):
        c = enc32_itoc[bits2int(bits[i:i+5])]
        if c == '\0':
            break
        text += c
    return text

# message should be bit string
# encoded should be text string
def expansion_ratio(message, encoded):
    message_bits = len(message)
    encoded_ba = bitarray.bitarray()
    encoded_ba.frombytes(encoded.encode('utf-8'))
    encoded_bits = len(encoded_ba.tolist())
    return encoded_bits/message_bits

def is_cit(enc, token, prev):
    # prev is list of token ids
    # returns True iff token is candidate-level inconsistent token (CIT)
    prev.append(token)
    temp_text = enc.tokenizer.decode(prev)
    prev_new = enc.tokenizer.encode(temp_text)
    return prev != prev_new
