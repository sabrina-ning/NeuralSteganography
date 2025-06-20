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
    finish_sent = True

    print("Initial Context:", context)
    context_tokens = encode_context(context, enc)
    # print(context_tokens)

    # ------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------

    # First encode message to uniform bits, without any context
    if unicode_enc:
        ba = bitarray.bitarray()
        ba.frombytes(message_str.encode('utf-8'))
        message = ba.tolist()
    else:
        message_ctx = enc.encode('<|endoftext|>')
        message_str += '<eos>'
        message = decode_arithmetic(model, enc, message_str, message_ctx, precision=40, topk=60000)
        # print(message)

    # Next encode bits into cover text, using arbitrary context
    out, nll, kl, words_per_bit, Hq = encode_arithmetic(model, enc, message, context_tokens, temp=temp, finish_sent=finish_sent, precision=precision, topk=topk)
    text = enc.decode(out)
    
    print("="*40 + " Original Message " + "="*40)
    print(message_str)
    # print(message)
    # print(len(message))

    print("="*40 + " Encoding " + "="*40)
    print(text)
    print('ppl: %0.2f, kl: %0.3f, words/bit: %0.2f, bits/word: %0.2f, entropy: %.2f' % (math.exp(nll), kl, words_per_bit, 1/words_per_bit, Hq/0.69315))
    
    # Decode binary message from bits using the same arbitrary context
    message_rec = decode_arithmetic(model, enc, text, context_tokens, temp=temp, precision=precision, topk=topk)
    
    print("="*40 + " Recovered Message " + "="*40)
    # print(message_rec)
    # print("=" * 80)

    # Finally map message bits back to original text
    if unicode_enc:
        message_rec = [bool(item) for item in message_rec]
        ba = bitarray.bitarray(message_rec)
        reconst = ba.tobytes().decode('utf-8', 'ignore')
    else:
        reconst = encode_arithmetic(model, enc, message_rec, message_ctx, precision=40, topk=60000)
        reconst = enc.decode(reconst[0])
    print(reconst)

    end = time.time()
    print(f"Took {end - start:.2f} sec")

    # # Testing >>>
    for i, (a, b) in enumerate(zip(message, message_rec)):
        if a != b:
            print(f"Bit mismatch at position {i}: {a} != {b}")
            break
    assert message == message_rec[:len(message)], "FAILED: bit mismatch"
    # print(len(message_str))
    # print(len(reconst))
    assert message_str == reconst[:len(message_str)], "FAILED: string mismatch"

    print()

def run_all_tests():
    start = time.time()
    model_name = "BAAI/Emu3-Stage1"
    enc, model = get_model(model_name=model_name)
    print("Successfully loaded model:", model_name)
    print(f"AutoModel: {type(model)}, AutoTokenizer: {type(enc)}\n")

    # 1) control
    # 2) \n at end
    # 3) space + two \n's in middle
    # 4) space + two \n's in middle + space
    # 5) space + two newlines in middle + space + two \t's in middle
    # 6) two newlines in middle + space + \n + space + two \n's at end
    # 7) empty

    messages = [
        "",
        "This is a very secret message?!",
        # "unpredictable\n\n\t\u0120definitely-un usual_characters!!!",
        # "This is an <eos> unusual use case.",
        # "A" * 1000,
        # "Washington received his initial military training and command with the Virginia Regiment during the French and Indian War. He was later elected to the Virginia House of Burgesses and was named a delegate to the Continental Congress, where he was appointed Commanding General of the nation's Continental Army. Washington led American forces, allied with France, in the defeat of the British at Yorktown. Once victory for the United States was in hand in 1783, Washington resigned his commission."
    ]

    print(f"Processing {len(messages)} messages")
    for message_str in messages:
        print('1')
        context = "Washington received his initial military training and command with the Virginia Regiment during the French and Indian War. He was later elected to the Virginia House of Burgesses and was named a delegate to the Continental Congress, where he was appointed Commanding General of the nation's Continental Army. Washington led American forces, allied with France, in the defeat of the British at Yorktown. Once victory for the United States was in hand in 1783, Washington resigned his commission."
        test_arithmetic(message_str, context, model, enc)

        print('2')
        context = "Washington received his initial military training and command with the Virginia Regiment during the French and Indian War. He was later elected to the Virginia House of Burgesses and was named a delegate to the Continental Congress, where he was appointed Commanding General of the nation's Continental Army. Washington led American forces, allied with France, in the defeat of the British at Yorktown. Once victory for the United States was in hand in 1783, Washington resigned his commission.\n"
        test_arithmetic(message_str, context, model, enc)

        print('3')
        context = "Washington received his initial military training and command with the Virginia Regiment during the French and Indian War. \n\nHe was later elected to the Virginia House of Burgesses and was named a delegate to the Continental Congress, where he was appointed Commanding General of the nation's Continental Army. Washington led American forces, allied with France, in the defeat of the British at Yorktown. Once victory for the United States was in hand in 1783, Washington resigned his commission."
        test_arithmetic(message_str, context, model, enc)

        print('4')
        context = "Washington received his initial military training and command with the Virginia Regiment during the French and Indian War. \n\n He was later elected to the Virginia House of Burgesses and was named a delegate to the Continental Congress, where he was appointed Commanding General of the nation's Continental Army. Washington led American forces, allied with France, in the defeat of the British at Yorktown. Once victory for the United States was in hand in 1783, Washington resigned his commission."
        test_arithmetic(message_str, context, model, enc)

        print('5')
        context = \
            """Washington received his initial military training and command with the Virginia Regiment during the French and Indian War. 

He was later elected to the Virginia House of Burgesses and was named a delegate to the Continental Congress, where he was appointed Commanding General of the nation's Continental Army. Washington led American forces, allied with France, in the defeat of the British at Yorktown. \t\tOnce victory for the United States was in hand in 1783, Washington resigned his commission."""
        test_arithmetic(message_str, context, model, enc)

        print('6')
        context = \
            """Washington received his initial military training and command with the Virginia Regiment during the French and Indian War.

 He was later elected to the Virginia House of Burgesses and was named a delegate to the Continental Congress, where he was appointed Commanding General of the nation's Continental Army. Washington led American forces, allied with France, in the defeat of the British at Yorktown. Once victory for the United States was in hand in 1783, Washington resigned his commission... \n \n\n"""
        test_arithmetic(message_str, context, model, enc)

        print('7')
        context = ""
        test_arithmetic(message_str, context, model, enc)

    end = time.time()
    print(f"All tests took {end - start:.2f} sec")
    print("Done.")


def test_encode(message_str, context, model, enc):
    ## PARAMETERS
    temp = 0.9
    precision = 26
    topk = 300
    finish_sent = False
    
    print("="*40 + " Encoding " + "="*40)
    print("Context:", context)
    print("Original message:", message_str)

    context_tokens = encode_context(context, enc)

    empty_ctx = enc.encode('<|endoftext|>')
    message_str += '<eos>'
    message = decode_arithmetic(model, enc, message_str, empty_ctx, precision=40, topk=60000)
    # print(message) # bits
    
    out, _, _, _, _ = encode_arithmetic(model, enc, message, context_tokens, temp=temp, finish_sent=finish_sent, precision=precision, topk=topk)
    text = enc.decode(out)
    print("Cover text:", text)

    print()

def run_encode_tests():
    model_name="BAAI/Emu3-Gen-hf"
    enc, model = get_model(model_name=model_name)
    print("Successfully loaded model:", model_name)
    print()

    context = "Washington received his initial military training and command with the Virginia Regiment during the French and Indian War. He was later elected to the Virginia House of Burgesses and was named a delegate to the Continental Congress, where he was appointed Commanding General of the nation's Continental Army. Washington led American forces, allied with France, in the defeat of the British at Yorktown. Once victory for the United States was in hand in 1783, Washington resigned his commission."
    
    message_str = ""
    test_encode(message_str, context, model, enc)

    message_str = "This is a very secret message?!"
    test_encode(message_str, context, model, enc)


def test_decode(cover_text, context, model, enc):
    ## PARAMETERS
    temp = 0.9
    precision = 26
    topk = 300
    
    print("="*40 + " Decoding " + "="*40)
    print("Context:", context)
    print("Cover text:", cover_text)

    context_tokens = encode_context(context, enc)

    message_rec = decode_arithmetic(model, enc, cover_text, context_tokens, temp=temp, precision=precision, topk=topk)
    print(message_rec) # bits

    # empty_ctx = enc.encode('<|endoftext|>')
    # reconst = encode_arithmetic(model, enc, message_rec, empty_ctx, precision=40, topk=60000)
    # reconst = enc.decode(reconst[0])
    # print(reconst) # text
    
    print()

def run_decode_tests():
    model_name="BAAI/Emu3-Gen-hf"
    enc, model = get_model(model_name=model_name)
    print("Successfully loaded model:", model_name)
    print()

    context = "Washington received his initial military training and command with the Virginia Regiment during the French and Indian War. He was later elected to the Virginia House of Burgesses and was named a delegate to the Continental Congress, where he was appointed Commanding General of the nation's Continental Army. Washington led American forces, allied with France, in the defeat of the British at Yorktown. Once victory for the United States was in hand in 1783, Washington resigned his commission."
    # cover_text = " His signature is found on the Treaty of Paris, which ended the Revolutionary War in 1783.The American forces under Washington under the command of Nathanael Greene were"
    # test_decode(cover_text, context, model, enc)

    cover_text = \
""" During his retirement he moved to Mount Vernon, which he then called home.
Washington died at age 67 on December 14, 1799 in New York City, and is buried in the cemetery at the Wallingford, Connecticut parish church where he was baptized.
Such is the story"""
    test_decode(cover_text, context, model, enc)


if __name__ == "__main__":
    # run_decode_tests()
    # run_encode_tests()
    run_all_tests()
