"""End-to-end neural image steganography pipeline using Emu3.

Encodes a secret image (or secret text) into the visual tokens of a cover image
using arithmetic coding conditioned on an arbitrary prompt context.
Optimizes cover image pixels using STE to guarantee exact token recovery through PNG save/load.
"""

import math
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoProcessor

from src.arithmetic_image import encode_arithmetic, decode_arithmetic
from src.arithmetic_image_to_bits import encode_arithmetic_from_bits, decode_arithmetic_to_bits
from src.optimize_pixels import optimize_pixels
from src.utils import get_model, encode_image, decode_image


def run_arithmetic(message_str, message_img_path, context, model, enc, unicode_enc=False):
    """Encodes a secret image or text message into a generated cover image using Emu3."""
    temp = 0.9
    precision = 26
    topk = 100
    finish_sent = False

    print("=" * 40 + " Context " + "=" * 40)
    print("context string:", context)

    inputs = enc(
        text=[context],
        padding=True,
        return_tensors="pt",
        return_for_image_generation=True,
    ).to(model.device)
    context_tokens = inputs["input_ids"][0]
    print("context tokens length:", len(context_tokens))

    print("=" * 40 + " Original Message " + "=" * 40)

    secret_model = model
    secret_enc = enc
    message_ctx = enc.tokenizer.encode("<|endoftext|>")
    secret_precision = 40

    if message_img_path:
        message_img_tokens, height, width, _ = encode_image(message_img_path, model, enc)
        message_tokens = message_img_tokens.tolist()
        print(f"message image tokens count: {len(message_tokens)}")
        message = decode_arithmetic_to_bits(
            secret_model, secret_enc, message_tokens, message_ctx,
            precision=secret_precision, topk=None)
    else:
        raise ValueError("message_img_path must be provided for image steganography.")

    print(f"num secret bits: {len(message)}")

    print("=" * 40 + " Encoding Cover Image " + "=" * 40)

    # Encode bits into cover image tokens using arbitrary prompt context
    out, _, _, _, _ = encode_arithmetic(
        model, enc, message, context_tokens,
        temp=temp, finish_sent=finish_sent, precision=precision, topk=topk)

    out = out[len(context_tokens):]

    # After stripping context, output is: [visual_tokens + eol markers] [eof] [eoi] [eos]
    eof_pos = (out == enc.tokenizer.eof_token_id).nonzero(as_tuple=True)[0][0].item()

    cover_image_tokens = out[:eof_pos]   # image tokens before eof
    n_image_tokens = len(cover_image_tokens)

    # Emu3-Gen outputs fixed 720x720 images
    downsample_ratio = enc.image_processor.spatial_factor
    img_h, img_w = 720, 720

    # Decode cover image tokens to raw pixels
    padded_tokens = torch.cat([cover_image_tokens, torch.zeros(3, dtype=cover_image_tokens.dtype, device=cover_image_tokens.device)])
    with torch.no_grad():
        decoded_pixels = model.model.decode_image_tokens(
            padded_tokens.unsqueeze(0),
            height=(img_h // downsample_ratio),
            width=(img_w // downsample_ratio))
    torch.cuda.empty_cache()

    # Save pre-optimization cover image for comparison
    raw_cover_image = enc.image_processor.postprocess(
        decoded_pixels.float(), return_tensors="PIL.Image.Image")["pixel_values"][0]
    raw_cover_path = f"output_images/{Path(message_img_path).stem}_cover_raw.png"
    raw_cover_image.save(raw_cover_path)
    print(f"Saved raw cover image to {raw_cover_path}")

    # Optimize pixels so they survive PNG round-trip with the same tokens
    print("=" * 40 + " Optimizing Cover Image " + "=" * 40)
    optimized_pixels = optimize_pixels(
        decoded_pixels, cover_image_tokens, img_h, img_w,
        base_model=model.model, image_processor=enc.image_processor, use_ste=True)

    # Save optimized cover image as PNG
    cover_image = enc.image_processor.postprocess(
        optimized_pixels.float(), return_tensors="PIL.Image.Image")["pixel_values"][0]
    cover_img_path = f"output_images/{Path(message_img_path).stem}_cover.png"
    cover_image.save(cover_img_path)
    print(f"Saved optimized cover image to {cover_img_path}")

    # Re-encode from saved PNG to get round-tripped tokens
    reenc_tokens, _, _, _ = encode_image(cover_img_path, model, enc)

    # Verify token recovery
    recovered_image_tokens = reenc_tokens[:n_image_tokens]
    match_count = int((recovered_image_tokens == cover_image_tokens).sum().item())
    print(f"Token recovery after PNG round-trip: {match_count}/{n_image_tokens}")

    out = reenc_tokens

    print("=" * 40 + " Decoding Secret Bits " + "=" * 40)

    # Decode binary message from bits using the same arbitrary context
    message_rec = decode_arithmetic(model, enc, out, context_tokens, temp=temp, precision=precision, topk=topk)
    print(f"Recovered {len(message_rec)} secret bits")

    # Reverse stage 2 then stage 1 with Emu3 backend
    reconst, _, _, _, _ = encode_arithmetic_from_bits(
        secret_model, secret_enc, message_rec, message_ctx,
        precision=secret_precision, topk=None)

    image_tokens = reconst
    print(f"Recovered secret image tokens: {len(image_tokens)}")

    recovered_image = decode_image(image_tokens, height, width, model, enc)
    recovered_path = "output_images/recovered.png"
    recovered_image.save(recovered_path)
    print(f"Recovered secret image saved to {recovered_path}")


def run_all_tests(model_name="BAAI/Emu3-Gen-hf"):
    start = time.time()
    enc, model = get_model(model_name=model_name)

    print("Successfully loaded:", model_name)
    print(f"Model: {type(model)}")
    print(f"Processor: {type(enc)}")

    message_img = "test_images/yosemite_720x720.png"
    context = "a puppy"
    run_arithmetic("", message_img, context, model, enc)
    end = time.time()
    print(f"Pipeline took {end - start:.2f} seconds")


if __name__ == "__main__":
    run_all_tests("BAAI/Emu3-Gen-hf")
