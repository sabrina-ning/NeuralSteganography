import math
import time
import ipdb

from arithmetic import encode_arithmetic, decode_arithmetic
from utils import get_model, encode_context, encode_image, decode_image

def test_arithmetic(message_str, message_img_path, context, model, enc, unicode_enc=False):
    start = time.time()
    # ipdb.set_trace()

    ## PARAMETERS
    temp = 0.9
    precision = 26
    topk = 300
    finish_sent = False

    print("="*40 + " Context " + "="*40)
    print("context string:", context)
    context_tokens = encode_context(context, enc)
    print("context tokens:", context_tokens)

    # ------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------

    print("="*40 + " Original Message " + "="*40)

    # First encode message to uniform bits, without any context
    message_ctx = enc.tokenizer.encode('<|endoftext|>')
    message_str += '<eos>'

    if message_img_path: # text + image
        message_img_tokens, height, width, _ = encode_image(message_img_path, model, enc)
        message_str_tokens = enc.tokenizer.encode(message_str)
        message_tokens = message_img_tokens.tolist() + [enc.tokenizer.bos_token_id] + message_str_tokens
        # message_tokens = message_img_tokens.tolist() + [151849] + message_str_tokens
        print("message tokens:", message_tokens)
        message = decode_arithmetic(model, enc, message_tokens, message_ctx, precision=50, topk=None)
    else: # text-only
        print('text-only!')
        message = decode_arithmetic(model, enc, message_str, message_ctx, precision=40, topk=None)
    print(f"\n[{message_str}]\n")
    # print("message bits:", message)
    print("num message bits:", len(message))

    print("="*40 + " Encoding " + "="*40)

    # Next encode bits into cover text, using arbitrary context
    out, nll, kl, words_per_bit, Hq = encode_arithmetic(model, enc, message, context_tokens, temp=temp, finish_sent=finish_sent, precision=precision, topk=topk)
    text = enc.tokenizer.decode(out)
    print("cover text:", text)
    print('ppl: %0.2f, kl: %0.3f, words/bit: %0.2f, bits/word: %0.2f, entropy: %.2f' % (math.exp(nll), kl, words_per_bit, 1/words_per_bit, Hq/0.69315))

    end = time.time()
    print(f"Encoding: Took {end - start:.2f} sec\n")

    breakpoint()

    start = time.time()

    # Decode binary message from bits using the same arbitrary context
    message_rec = decode_arithmetic(model, enc, text, context_tokens, temp=temp, precision=precision, topk=topk)
    
    print("="*40 + " Recovered Message " + "="*40)
    print("recovered message bits:", message_rec)

    # Finally map message bits back to original text
    reconst, _, _, _, _ = encode_arithmetic(model, enc, message_rec, message_ctx, precision=50, topk=None)

    # Check if message has image
    if reconst[0] == enc.tokenizer.image_wrapper_token_id:
    # if reconst[0] == 151851:
        print('has image!')
        last_end_token_idx = reconst.index(enc.tokenizer.eos_token_id)
        # last_end_token_idx = reconst.index(151850)
        image_tokens = reconst[1:last_end_token_idx + 1]
        print("recovered image tokens:", image_tokens)

        image = decode_image(image_tokens, height, width, model, enc)
        image.save(message_img_path + ".jpg")

        print("recovered text tokens:", reconst[last_end_token_idx + 2:]) # skip BOS
        reconst = enc.tokenizer.decode(reconst[last_end_token_idx + 2:])
    else: # text-only
        print("recovered text tokens:", reconst)
        reconst = enc.tokenizer.decode(reconst)
    print(f"\n[{reconst}]\n")

    end = time.time()
    print(f"Decoding: Took {end - start:.2f} sec")

    # Testing >>>
    # for i, (a, b) in enumerate(zip(message, message_rec)):
    #     if a != b:
    #         print(f"Bit mismatch at position {i}: {a} != {b}")
    #         break
    # assert message == message_rec[:len(message)], "FAILED: bit mismatch"
    # assert message_str == reconst[:len(message_str)], "FAILED: string mismatch"

    # if message_str != reconst[:len(message_str)]:
    #     print("FAILED: string mismatch")

    # if message != message_rec[:len(message)]:
    #     print("FAILED: bit mismatch")
    #     exit

    print()

def run_all_tests(model_name):
    enc, model = get_model(model_name=model_name)
    print("Successfully loaded:", model_name)
    print(f"Model: {type(model)}")
    print(f"Processor: {type(enc)}")
    print(f"Tokenizer: {type(enc.tokenizer)}")

    messages = [
        "This is a very secret message!",
        "The quick brown fox jumps over the lazy dog.",
        # "The password to this account is: abcDEF01234"
    ]

    images = [
        "images/cat_64.jpg",
        "images/cornell_64.jpg",
        "images/yosemite_64.jpg"
    ]

    for i, message in enumerate(messages):
        print(f"Message {i+1}")

        for j, image in enumerate(images):
            print(f"Image {j+1}")

            # print('1')
            # context = """Instagram is an American photo and short-form video sharing social networking service owned by Meta Platforms. It allows users to upload media that can be edited with filters, be organized by hashtags, and be associated with a location via geographical tagging. Posts can be shared publicly or with preapproved followers."""
            # test_arithmetic(message, image, context, model, enc)

            # print('2')
            # context = """Cornell University is a private Ivy League research university based in Ithaca, New York, United States. The university was co-founded by American philanthropist Ezra Cornell and historian and educator Andrew Dickson White in 1865. Since its founding, Cornell University has been a co-educational and nonsectarian institution."""
            # test_arithmetic(message, image, context, model, enc)

#             print('3')
#             context = """Abraham Lincoln (February 12, 1809 – April 15, 1865) was the 16th president of the United States, serving from 1861 until his assassination in 1865. He led the United States through the American Civil War, defeating the Confederate States of America and playing a major role in the abolition of slavery.
# Lincoln was born into poverty in Kentucky and raised on the frontier."""
#             test_arithmetic(message, image, context, model, enc)

            # print('4')
            # context = """San Francisco, officially the City and County of San Francisco, is a commercial, financial, and cultural center of Northern California. With a population of 827,526 residents as of 2024, San Francisco is the fourth-most populous city in the U.S. state of California and the 17th-most populous in the United States. San Francisco has a land area of 46.9 square miles (121 square kilometers) at the upper end of the San Francisco Peninsula and is the fifth-most densely populated U.S. county."""
            # test_arithmetic(message, image, context, model, enc)

            print('5')
            context = """Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability with the use of significant indentation.

Python is dynamically type-checked and garbage-collected. It supports multiple programming paradigms, including structured (particularly procedural), object-oriented and functional programming."""
            test_arithmetic(message, image, context, model, enc)

    print("Done.")    


if __name__ == "__main__":
    run_all_tests("BAAI/Emu3-Chat-hf")
    # run_all_tests("BAAI/Emu3-Gen-hf")
