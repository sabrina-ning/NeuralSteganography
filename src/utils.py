"""Shared utility functions for neural steganography.

Includes:
- Model/processor loading (Emu3, GPT-2, Qwen)
- Image encoding/decoding via Emu3 VQ tokenizer
- Arithmetic coding primitives (bit/int conversion, KL, entropy)
- Simple 32-character text encoding for demo messages
"""

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


def limit_past(past, max_len=1022):
    """Truncate KV cache to at most `max_len` tokens.

    Handles both old-style (list of tensors) and new-style (list of tuples)
    cache formats from HuggingFace transformers.
    """
    past = list(past)
    for i in range(len(past)):
        if isinstance(past[i], tuple):
            key, value = past[i]
            past[i] = (
                key[:, :, -max_len:, :],
                value[:, :, -max_len:, :]
            )
        else:
            past[i] = past[i][:, :, -max_len:, :]
    return tuple(past)


def kl(q, logq, logp):
    """KL divergence D(q || p) in bits."""
    res = q * (logq - logp) / 0.69315
    res[q == 0] = 0
    return res.sum().item()


def entropy(q, logq):
    """Shannon entropy H(q) in bits."""
    res = q * logq / 0.69315
    res[q == 0] = 0
    return -res.sum().item()


def bits2int(bits):
    """Convert a list of bits to an integer. E.g. [0, 1, 1, 1] -> 14."""
    res = 0
    for i, bit in enumerate(bits):
        res += bit * (2 ** i)
    return res


def int2bits(inp, num_bits):
    """Convert an integer to a list of `num_bits` bits (LSB first)."""
    if num_bits == 0:
        return []
    strlist = ('{0:0%db}' % num_bits).format(inp)
    return [int(strval) for strval in reversed(strlist)]


def is_sent_finish(token_idx, enc):
    """Check if a token represents a sentence-ending punctuation."""
    token = enc.tokenizer.decode([token_idx])
    return '.' in token or '!' in token or '?' in token


def num_same_from_beg(bits1, bits2):
    """Count the number of matching bits from the beginning of two bit lists."""
    assert len(bits1) == len(bits2)
    for i in range(len(bits1)):
        if bits1[i] != bits2[i]:
            break
    return i


def encode_context(raw_text, enc):
    """Encode a text string into context tokens with an end-of-text prefix."""
    context_tokens = enc.tokenizer.encode('<|endoftext|>') + enc.tokenizer.encode(raw_text)
    return context_tokens


def encode_image(image_path, model, enc):
    """Encode an image file into Emu3 BPE image tokens.

    Returns:
        image_tokens: 1-D tensor of BPE tokens (with EOL markers + trailing eof/eoi/eos)
        height: original image height in pixels
        width: original image width in pixels
        image_PIL: the loaded PIL Image object
    """
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    image_PIL = image

    enc.image_processor.do_resize = False
    image = enc.image_processor.preprocess(image, return_tensors="pt")
    pixel_values = image["pixel_values"].to(torch.bfloat16).cuda()
    image_sizes = image["image_sizes"].cuda()

    with torch.no_grad():
        image_tokens = model.model.get_image_tokens(pixel_values, image_sizes).cuda()

    end_tokens = torch.tensor([enc.tokenizer.eof_token_id, enc.tokenizer.eoi_token_id, enc.tokenizer.eos_token_id], device="cuda")
    image_tokens = torch.cat([image_tokens, end_tokens])

    return image_tokens, height, width, image_PIL


def decode_image(image_tokens, height, width, model, enc):
    """Decode Emu3 BPE image tokens back into a PIL Image.

    Args:
        image_tokens: 1-D tensor of BPE tokens (without trailing eof/eoi/eos)
        height: image height in pixels
        width: image width in pixels
    """
    if not isinstance(image_tokens, torch.Tensor):
        image_tokens = torch.tensor(image_tokens, device="cuda")
    image = model.model.decode_image_tokens(
        image_tokens.unsqueeze(0),
        height=(height // enc.image_processor.spatial_factor),
        width=(width // enc.image_processor.spatial_factor))
    image = enc.image_processor.postprocess(image, return_tensors="PIL.Image.Image")['pixel_values'][0]
    return image


def get_model(seed=1234, model_name='gpt2'):
    """Load a language model and its processor/tokenizer.

    Supports GPT-2 variants, Qwen, and Emu3 (both -hf and original checkpoints).

    Returns:
        enc: the processor (for Emu3) or tokenizer (for text-only models)
        model: the loaded model in eval mode
    """
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    enc = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map=device)
    enc.image_processor.min_pixels = 256 * 256

    if "hf" in model_name:
        model = Emu3ForConditionalGeneration.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map=device)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map=device)

    model.eval()
    return enc, model


# --- Simple 32-character text encoding for demo messages ---

enc32_itoc = ['\0', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k',
              'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w',
              'x', 'y', 'z', '.', ',', "'", '!', ' ']
enc32_ctoi = {k: v for v, k in enumerate(enc32_itoc)}


def enc32(text):
    """Encode text using a simple 32-character alphabet (5 bits per char)."""
    bits = []
    for c in text:
        bits.extend(int2bits(enc32_ctoi[c], 5))
    return bits


def dec32(bits):
    """Decode bits back to text using the 32-character alphabet."""
    text = ''
    for i in range(0, len(bits), 5):
        c = enc32_itoc[bits2int(bits[i:i + 5])]
        if c == '\0':
            break
        text += c
    return text


def expansion_ratio(message, encoded):
    """Compute the bit expansion ratio between a message and its encoded form."""
    message_bits = len(message)
    encoded_ba = bitarray.bitarray()
    encoded_ba.frombytes(encoded.encode('utf-8'))
    encoded_bits = len(encoded_ba.tolist())
    return encoded_bits / message_bits


def is_cit(enc, token, prev):
    """Check if a token is a candidate-level inconsistent token (CIT).

    A CIT is a token that, when appended to prev and re-tokenized,
    produces a different token sequence. This happens due to tokenizer
    merge rules (e.g., whitespace + letter merging into a single token).
    """
    prev.append(token)
    temp_text = enc.tokenizer.decode(prev)
    prev_new = enc.tokenizer.encode(temp_text)
    return prev != prev_new
