import bitarray
import math
import time

from arithmetic import encode_arithmetic, decode_arithmetic
from utils import get_model, encode_context, encode

def test_arithmetic(message_str, context, model, enc, unicode_enc=False):
    start = time.time()

    ## PARAMETERS
    temp = 0.9
    precision = 26
    topk = 300
    finish_sent = False

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
        # message_ctx = encode(enc, '<|endoftext|>')
        # print(message_ctx)
        message_str += '<eos>'
        message = decode_arithmetic(model, enc, message_str, message_ctx, precision=40, topk=60000)

    print("="*40 + " Original Message " + "="*40)
    print(message_str)
    # print(message)
    print(len(message))

    # Next encode bits into cover text, using arbitrary context
    out, nll, kl, words_per_bit, Hq = encode_arithmetic(model, enc, message, context_tokens, temp=temp, finish_sent=finish_sent, precision=precision, topk=topk)
    text = enc.decode(out)

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
    print(len(message_rec))

    end = time.time()
    print(f"Took {end - start:.2f} sec")

    # Testing >>>
    # for i, (a, b) in enumerate(zip(message, message_rec)):
    #     if a != b:
    #         print(f"Bit mismatch at position {i}: {a} != {b}")
    #         break
    # assert message == message_rec[:len(message)], "FAILED: bit mismatch"
    # assert message_str == reconst[:len(message_str)], "FAILED: string mismatch"
    if message_str != reconst[:len(message_str)]:
        print("FAILED: string mismatch")

    print()

def run_all_tests(model_name):
    enc, model = get_model(model_name=model_name)
    print("Successfully loaded model:", model_name)
    print(f"AutoModel: {type(model)}, AutoTokenizer: {type(enc)}")

    message_strs = [
        "",
        "This is a very secret message!",
        "unpredictable\n\n\tdefinitely-un usual_characters!!!",
        "A" * 100,
        "endoftext",
        "Special symbols !@#$%^&*()_+-=[]{}|;':\",./<>?"
    ]

    for message_str in message_strs:
        print('1')
        context = "Washington received his initial military training and command with the Virginia Regiment during the French and Indian War. He was later elected to the Virginia House of Burgesses and was named a delegate to the Continental Congress, where he was appointed Commanding General of the nation's Continental Army. Washington led American forces, allied with France, in the defeat of the British at Yorktown. Once victory for the United States was in hand in 1783, Washington resigned his commission."
        test_arithmetic(message_str, context, model, enc)

        print('2')
        context = """Washington received his initial military training and command with the Virginia Regiment during the French and Indian War. 

He was later elected to the Virginia House of Burgesses and was named a delegate to the Continental Congress, where he was appointed Commanding General of the nation's Continental Army. \n\nWashington led American forces, allied with France, in the defeat of the British at Yorktown. Once victory for the United States was in hand in 1783, Washington resigned his commission."""
        test_arithmetic(message_str, context, model, enc)

        print('3')
        context = """Washington received his initial military training and command with the Virginia Regiment during the French and Indian War. 

He was later elected to the Virginia House of Burgesses and was named a delegate to the Continental Congress, where he was appointed Commanding General of the nation's Continental Army. \n\n
Washington led American forces, allied with France, in the defeat of the British at Yorktown. Once victory for the United States was in hand in 1783, Washington resigned his commission."""
        test_arithmetic(message_str, context, model, enc)

        print('4')
        context = "Cornell University is a private Ivy League research university based in Ithaca, New York, United States. The university was "
        test_arithmetic(message_str, context, model, enc)

        print('5')
        context = """Cornell University is a private Ivy League research university based in Ithaca, New York, United States. 
The university was """
        test_arithmetic(message_str, context, model, enc)

        print('6')
        context = "Cornell University is a private[4] Ivy League research university based in Ithaca, New York, United States. The university was co-founded by American philanthropist Ezra Cornell and historian and educator Andrew Dickson White in 1865. Since its founding,"
        test_arithmetic(message_str, context, model, enc)

    print("Done.")


if __name__ == "__main__":
    # run_all_tests("gpt2-medium")
    # run_all_tests("BAAI/Emu3-Stage1")
    # run_all_tests("BAAI/Emu3-Chat")
    run_all_tests("BAAI/Emu3-Chat-hf")
    # run_all_tests("BAAI/Emu3-Gen")
