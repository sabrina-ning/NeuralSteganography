import bitarray
import math
import time

from arithmetic import encode_arithmetic, decode_arithmetic
from utils import get_model, encode_context

def test_arithmetic(message_str, context, model, enc, unicode_enc=False):
    start = time.time()

    ## PARAMETERS
    temp = 0.9
    precision = 26
    topk = 300
    finish_sent = False

    print("="*40 + " Context " + "="*40)
    print(f"[{context}]")
    print()
    context_tokens = encode_context(context, enc)
    print("context tokens >>>", context_tokens)

    # ------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------

    print("="*40 + " Original Message " + "="*40)

    # First encode message to uniform bits, without any context
    if unicode_enc:
        ba = bitarray.bitarray()
        ba.frombytes(message_str.encode('utf-8'))
        message = ba.tolist()
    else:
        message_ctx = enc.encode('<|endoftext|>')
        message_str += '<eos>'
        print(f"[{message_str}]")
        print()
        message = decode_arithmetic(model, enc, message_str, message_ctx, precision=40, topk=None)
        print()
    print(message)

    print("="*40 + " Encoding -> Cover Text " + "="*40)

    # Next encode bits into cover text, using arbitrary context
    out, nll, kl, words_per_bit, Hq = encode_arithmetic(model, enc, message, context_tokens, temp=temp, finish_sent=finish_sent, precision=precision, topk=topk)
    text = enc.decode(out)
    print()
    print(f"[{text}]")
    print()
    print('ppl: %0.2f, kl: %0.3f, words/bit: %0.2f, bits/word: %0.2f, entropy: %.2f' % (math.exp(nll), kl, words_per_bit, 1/words_per_bit, Hq/0.69315))
    print()

    # Decode binary message from bits using the same arbitrary context
    message_rec = decode_arithmetic(model, enc, text, context_tokens, temp=temp, precision=precision, topk=topk)
    
    print("="*40 + " Recovered Message " + "="*40)
    print(message_rec)
    print()

    # Finally map message bits back to original text
    if unicode_enc:
        message_rec = [bool(item) for item in message_rec]
        ba = bitarray.bitarray(message_rec)
        reconst = ba.tobytes().decode('utf-8', 'ignore')
    else:
        reconst = encode_arithmetic(model, enc, message_rec, message_ctx, precision=40, topk=None)
        reconst = enc.decode(reconst[0])
        print(f"[{reconst}]")
        print()

    print(message_rec)
    print()

    end = time.time()
    print(f"Took {end - start:.2f} sec")

    # Testing >>>
    # for i, (a, b) in enumerate(zip(message, message_rec)):
    #     if a != b:
    #         print(f"Bit mismatch at position {i}: {a} != {b}")
    #         break
    # assert message == message_rec[:len(message)], "FAILED: bit mismatch"
    # assert message_str == reconst[:len(message_str)], "FAILED: string mismatch"

    # if message_str != reconst[:len(message_str)]:
    #     print("FAILED: string mismatch")

    if message != message_rec[:len(message)]:
        print("FAILED: bit mismatch")
        exit

    print()

def run_all_tests(model_name):
    enc, model = get_model(model_name=model_name)
    print("Successfully loaded:", model_name)
    print(f"Model: {type(model)}")
    print(f"Tokenizer: {type(enc)}")

    messages = [
        "",
        "This is a very secret message!"
    ]

    # messages = [
    #     [0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1],
    #     [0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0],
    #     [1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1],
    #     [1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0]
    # ]

    # messages = [
    #     [1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1],
    #     [0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0],
    #     [1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
    #     [1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1],
    # ]

    for i, message in enumerate(messages):
        print(f"Message {i+1}\n")

        print('1')
        context = """Instagram is an American photo and short-form video sharing social networking service owned by Meta Platforms. It allows users to upload media that can be edited with filters, be organized by hashtags, and be associated with a location via geographical tagging. Posts can be shared publicly or with preapproved followers."""
        test_arithmetic(message, context, model, enc)

        print('2')
        context = """Cornell University is a private Ivy League research university based in Ithaca, New York, United States. The university was co-founded by American philanthropist Ezra Cornell and historian and educator Andrew Dickson White in 1865. Since its founding, Cornell University has been a co-educational and nonsectarian institution."""
        test_arithmetic(message, context, model, enc)

        print('3')
        context = """Abraham Lincoln (February 12, 1809 – April 15, 1865) was the 16th president of the United States, serving from 1861 until his assassination in 1865. He led the United States through the American Civil War, defeating the Confederate States of America and playing a major role in the abolition of slavery.
Lincoln was born into poverty in Kentucky and raised on the frontier."""
        test_arithmetic(message, context, model, enc)

        print('4')
        context = """San Francisco, officially the City and County of San Francisco, is a commercial, financial, and cultural center of Northern California. With a population of 827,526 residents as of 2024, San Francisco is the fourth-most populous city in the U.S. state of California and the 17th-most populous in the United States. San Francisco has a land area of 46.9 square miles (121 square kilometers) at the upper end of the San Francisco Peninsula and is the fifth-most densely populated U.S. county."""
        test_arithmetic(message, context, model, enc)

        print('5')
        context = """Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability with the use of significant indentation.
Python is dynamically type-checked and garbage-collected. It supports multiple programming paradigms, including structured (particularly procedural), object-oriented and functional programming.

"""
        test_arithmetic(message, context, model, enc)

        print('6')
        context = """Users can browse other users' content by tags and locations, view trending content, like photos, and follow other users to add their content to a personal feed.[10] A Meta-operated image-centric social media platform, it is available on iOS, Android, Windows 10, and the web. Users can take photos and edit them using built-in filters and other tools, then share them on other social media platforms like Facebook."""
        test_arithmetic(message, context, model, enc)

        print('7')
        context = """As of fall 2024, the student body included 16,128 undergraduate and 10,665 graduate students from all 50 U.S. states and 130 countries.[7]

The university is organized into eight undergraduate colleges and seven graduate divisions on its main Ithaca campus.[12] Each college and academic division has near autonomy in defining its respective admission standards and academic curriculum."""
        test_arithmetic(message, context, model, enc)

        print('8')
        context = """He was self-educated and became a lawyer, Illinois state legislator, and U.S. representative. Angered by the Kansas–Nebraska Act of 1854, which opened the territories to slavery, he became a leader of the new Republican Party. He reached a national audience in the 1858 Senate campaign debates against Stephen A. Douglas."""
        test_arithmetic(message, context, model, enc)

        print('9')
        context = """Among U.S. cities proper with over 250,000 residents, San Francisco is ranked first by per capita income and sixth by aggregate income as of 2023.[25] San Francisco anchors the 13th-most populous metropolitan statistical area in the U.S., with almost 4.6 million residents in 2023. The larger San Jose–San Francisco–Oakland combined statistical area, the fifth-largest urban region in the U.S., had a 2023 estimated population of over nine million."""
        test_arithmetic(message, context, model, enc)

        print('10')
        context = """Guido van Rossum began working on Python in the late 1980s as a successor to the ABC programming language, and he first released it in 1991 as Python 0.9.0.[34] Python 2.0 was released in 2000. Python 3.0, released in 2008, was a major revision not completely backward-compatible with earlier versions. Python 2.7.18, released in 2020, was the last release of Python 2.[35]"""
        test_arithmetic(message, context, model, enc)

    print("Done.")


if __name__ == "__main__":
    # run_all_tests("gpt2")
    # run_all_tests("BAAI/Emu3-Stage1")
    # run_all_tests("BAAI/Emu3-Gen")

    run_all_tests("BAAI/Emu3-Chat-hf")
    # run_all_tests("BAAI/Emu3-Chat")
