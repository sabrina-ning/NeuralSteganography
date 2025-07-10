from PIL import Image
import torch
import numpy as np
import bitarray

from transformers import Emu3ForCausalLM, AutoModelForCausalLM, AutoTokenizer, AutoModel, AutoImageProcessor, Emu3ForConditionalGeneration, Emu3Processor, AutoProcessor, GPT2TokenizerFast

def decode(self, token_ids, **kwargs):
    filtered_tokens = self.convert_ids_to_tokens(token_ids)
    text = self.convert_tokens_to_string(filtered_tokens)
    return text
GPT2TokenizerFast.decode = decode

def _convert_token_to_id(self, token):
    return self.encoder.get(token, 0)
GPT2TokenizerFast._convert_token_to_id = _convert_token_to_id


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
    token = enc.decode([token_idx])
    return '.' in token or '!' in token or '?' in token

def num_same_from_beg(bits1, bits2):
    assert len(bits1) == len(bits2)
    for i in range(len(bits1)):
        if bits1[i] != bits2[i]:
            break
    return i

def encode(processor, text):
    encoded = processor(text=text, return_tensors="pt")
    return encoded['input_ids'][0].tolist()

def encode_context(raw_text, enc):
    context_tokens = enc.encode('<|endoftext|>') + enc.encode(raw_text)
    # context_tokens = encode(enc, '<|endoftext|>') + encode(enc, raw_text)
    return context_tokens

# Use gpt2-medium for 345M param model
# Use gpt2-large for 774M param model
def get_model(seed=1234, model_name='gpt2'):
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if "hf" in model_name:
        enc = AutoTokenizer.from_pretrained(
            "BAAI/Emu3-Chat",
            trust_remote_code=True,
            device_map="cuda:0")
        model = Emu3ForConditionalGeneration.from_pretrained(
            model_name, 
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="cuda:0")
    else:
        enc = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map="cuda:0")
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="cuda:0")
    # model.to(device)
    model.eval()
    # model.double()

    return enc, model

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

# converts raw image into tokens in LM vocab
def encode_image(image_path, vision_enc, image_proc):
    image = Image.open(image_path).convert("RGB")
    image_tensors = image_proc(image, return_tensors="pt")["pixel_values"].to(torch.float16).cuda()
    print(image_tensors.shape) # [1, 3, 512, 512]
    with torch.no_grad():
        visual_tokens = vision_enc.encode(image_tensors)
        print(visual_tokens.shape) # [1, 64, 64]
    tokens = visual_tokens + 151854 # offset by ID of visual token 0
    return tokens

# converts tokens in LM vocab into raw image and saves to image path
def decode_image(tokens, image_path, vision_enc, image_proc):
    visual_tokens = tokens - 151854 # undo offset by ID of visual token 0
    with torch.no_grad():
        recon = vision_enc.decode(visual_tokens.view(1, 16, 16))
    recon = recon.view(-1, *recon.shape[2:])
    recon_image = image_proc.postprocess(recon)["pixel_values"][0]
    recon_image.save(image_path)

def get_vision_tokenizer(model_name="BAAI/Emu3-VisionTokenizer"):
    image_proc = AutoImageProcessor.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map="cuda:0")
    print("Loaded image processor")
    
    vision_enc = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="cuda:0").eval()
    print("Loaded vision tokenizer")
    
    return image_proc, vision_enc