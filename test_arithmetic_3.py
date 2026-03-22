import math
from pathlib import Path

from arithmetic_image import encode_arithmetic, decode_arithmetic
from arithmetic_image_to_bits import encode_arithmetic_from_bits, decode_arithmetic_to_bits
from utils import get_model, encode_context, encode_image, decode_image

from transformers import AutoModelForCausalLM, AutoModel, AutoImageProcessor, AutoTokenizer
import torch
import time

# encodes and decodes either a single text (message_str) or image (message_img_path) message, using context
def run_arithmetic(message_str, message_img_path, context, model, enc, unicode_enc=False):
    temp = 0.9
    precision = 26
    topk = 100
    finish_sent = False

    print("="*40 + " Context " + "="*40)

    print("context string:", context)
    # context_tokens = encode_context(context, enc)
    inputs = enc(
        text=[context],
        padding=True,
        return_tensors="pt",
        return_for_image_generation=True,
    ).to(model.device)
    context_tokens = inputs["input_ids"][0]
    print("context tokens:", context_tokens)

    print("="*40 + " Original Message " + "="*40)

    # First encode message to uniform bits, without any context
    message_ctx = enc.tokenizer.encode('<|endoftext|>')
    # message_str += '<eos>'

    if message_img_path: # image-only
        message_img_tokens, height, width, _ = encode_image(message_img_path, model, enc) # tokenize image
        message_tokens = message_img_tokens.tolist()
        print("message tokens:", message_tokens)
        print("# tokens:", len(message_tokens))
        message = decode_arithmetic_to_bits(model, enc, message_tokens, message_ctx, precision=60, topk=None)
    # else: # text-only
    #     message = decode_arithmetic(model, enc, message_str, message_ctx, precision=40, topk=None)

    print("message bits:", message)
    print("num bits:", len(message))

    print("="*40 + " Encoding " + "="*40)

    # Next encode bits into cover text, using arbitrary context
    out, _, _, _, _ = encode_arithmetic(model, enc, message, context_tokens, temp=temp, finish_sent=finish_sent, precision=precision, topk=topk)

    print("output tokens:", out)
    torch.save(out, "output_tokens.pt")
    # out = torch.load("output_tokens.pt")
    # hard coded for now >>>
    out = out[len(context_tokens):]
    # <<<
    image_new = decode_image(out, 720, 720, model, enc)
    cover_img_path = f"test_images/{Path(message_img_path).stem}_cover.png"
    image_new.save(cover_img_path)

    out, _, _, _ = encode_image(cover_img_path, model, enc)

    # Remove the image wrapper token (first token) before decoding
    # TODO: parse every k number of tokens as one image at a time, then decode each image iteratively
    # print(len(out))
    # for i in range(0, len(out), len(context_tokens) + 8193):
    #     image_tokens = out[i + len(context_tokens) : i + len(context_tokens) + 8193]
    #     cover_image = decode_image(image_tokens, 720, 720, model, enc)
    #     print(f"cover image {i//(len(context_tokens) + 8193)}:", cover_image)
    #     cover_image.save(f"new_images/cover_{i//(len(context_tokens) + 8193)}.png")

    print("="*40 + " Decoding " + "="*40)

    # Decode binary message from bits using the same arbitrary context
    # TODO: check if saving, loading, then encoding twice gives the same image tokens
    message_rec = decode_arithmetic(model, enc, out, context_tokens, temp=temp, precision=precision, topk=topk)
   
    print("="*40 + " Recovered Message " + "="*40)

    print("recovered message bits:", message_rec)

    # Finally map message bits back to original text
    reconst, _, _, _, _ = encode_arithmetic_from_bits(model, enc, message_rec, message_ctx, precision=60, topk=None)
    # breakpoint()

    # Check if message has image
    # if reconst[0][0] == enc.tokenizer.image_wrapper_token_id:
    eos_id = 151850 # end of sequence <|extra_204|>
    # end_index = reconst.index(eos_id)
    # image_tokens = reconst[:end_index + 1]
    image_tokens = reconst
    print("recovered message tokens:", image_tokens)

    image = decode_image(image_tokens, height, width, model, enc)
    image.save("new_images/recovered.png")
    # else: # text-only
    #     print("recovered text tokens:", reconst)
    #     reconst = enc.tokenizer.decode(reconst)
    print(f"\n[{reconst}]\n")


def run_all_tests(model_name):
    start = time.time()
    enc, model = get_model(model_name=model_name)

    print("Successfully loaded:", model_name)
    print(f"Model: {type(model)}")
    print(f"Processor: {type(enc)}")
    print(f"Tokenizer: {type(enc.tokenizer)}")

    message = ""
    image = "test_images/nature_256x256.png"
    context = "A kitten playing with a ball of yarn."
    run_arithmetic(message, image, context, model, enc)
    end = time.time()
    print(f"Took {end - start} seconds")


if __name__ == "__main__":
    run_all_tests("BAAI/Emu3-Gen-hf")
    # run_all_tests("BAAI/Emu3-Chat-hf")
