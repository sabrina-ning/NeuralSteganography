import bitarray
import math

from utils import get_model, encode_context

from arithmetic import encode_arithmetic, decode_arithmetic
from block_baseline import get_bins, encode_block, decode_block
from huffman_baseline import encode_huffman, decode_huffman
from sample import sample

def main():
    enc, model = get_model(model_name='gpt2')

    
    ## PARAMETERS
    # message_str = "This is a very secret message!"

    # FIXME: doesn't recover original message completely
    message_str = """Instagram[a] is an American photo and short-form video sharing social networking service owned by Meta Platforms. It allows users to upload media that can be edited with filters, be organized by hashtags, and be associated with a location via geographical tagging. Posts can be shared publicly or with preapproved followers. Users can browse other users' content by tags and locations, view trending content, like photos, and follow other users to add their content to a personal feed.[10] A Meta-operated image-centric social media platform, it is available on iOS, Android, Windows 10, and the web. Users can take photos and edit them using built-in filters and other tools, then share them on other social media platforms like Facebook. It supports 32 languages including English, Hindi, Spanish, French, Korean, and Japanese.[11]

Instagram was originally distinguished by allowing content to be framed only in a square (1:1) aspect ratio of 640 pixels to match the display width of the iPhone at the time. In 2015, this restriction was eased with an increase to 1080 pixels. It also added messaging features, the ability to include multiple images or videos in a single post, and a Stories feature—similar to its main competitor, Snapchat, which allowed users to post their content to a sequential feed, with each post accessible to others for 24 hours. As of January 2019, Stories was used by 500 million people daily.[10]

Instagram was launched for iOS in October 2010 by Kevin Systrom and Mike Krieger. It rapidly gained popularity, reaching 1 million registered users in two months, 10 million in a year, and 1 billion in June 2018. In April 2012, Facebook acquired the service for approximately US$1 billion in cash and stock. The Android version of Instagram was released in April 2012, followed by a feature-limited desktop interface in November 2012, a Fire OS app in June 2014, and an app for Windows 10 in October 2016. Although often admired for its success and influence, Instagram has also been criticized for negatively affecting teens' mental health, its policy and interface changes, its alleged censorship, and illegal and inappropriate content uploaded by users.

History
For a chronological guide, see Timeline of Instagram.
Instagram began development in San Francisco as Burbn, a mobile check-in app created by Kevin Systrom and Mike Krieger. On March 5, 2010, Systrom closed a $500,000 (equivalent to $682,200 in 2023) seed funding round with Baseline Ventures and Andreessen Horowitz while working on Burbn.[12][13] Realizing that it was too similar to Foursquare, they refocused their app on photo-sharing, which had become a popular feature among its users.[14][15] They renamed it Instagram, a portmanteau of instant camera and telegram.[16]"""

    unicode_enc = False
    mode = 'arithmetic'
    block_size = 3 # for huffman and bins
    temp = 0.9 # for arithmetic
    precision = 26 # for arithmetic
    sample_tokens = 100 # for sample
    topk = 300
    finish_sent=False # whether or not to force finish sent. If so, stats displayed will be for non-finished sentence

    ## VALIDATE PARAMETERS
    if mode not in ['arithmetic', 'huffman', 'bins', 'sample']:
        raise NotImplementedError

    if mode == 'bins':
        bin2words, words2bin = get_bins(len(enc.encoder), block_size)

    context = \
"""Washington received his initial military training and command with the Virginia Regiment during the French and Indian War. He was later elected to the Virginia House of Burgesses and was named a delegate to the Continental Congress, where he was appointed Commanding General of the nation's Continental Army. Washington led American forces, allied with France, in the defeat of the British at Yorktown. Once victory for the United States was in hand in 1783, Washington resigned his commission.


"""

    context_tokens = encode_context(context, enc)

    # ------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------

    # First encode message to uniform bits, without any context
    # (not essential this is arithmetic vs ascii, but it's more efficient when the message is natural language)
    if unicode_enc:
        ba = bitarray.bitarray()
        ba.frombytes(message_str.encode('utf-8'))
        message = ba.tolist()
    else:
        message_ctx = enc.encode('<|endoftext|>')
        message_str += '<eos>'
        message = decode_arithmetic(model, enc, message_str, message_ctx, precision=40, topk=60000)

    # Next encode bits into cover text, using arbitrary context
    Hq = 0
    if mode == 'arithmetic':
        out, nll, kl, words_per_bit, Hq = encode_arithmetic(model, enc, message, context_tokens, temp=temp, finish_sent=finish_sent, precision=precision, topk=topk)
    elif mode == 'huffman':
        out, nll, kl, words_per_bit = encode_huffman(model, enc, message, context_tokens, block_size, finish_sent=finish_sent)
    elif mode == 'bins':
        out, nll, kl, words_per_bit = encode_block(model, enc, message, context_tokens, block_size, bin2words, words2bin, finish_sent=finish_sent)
    elif mode == 'sample':
        out, nll, kl, Hq = sample(model, enc, sample_tokens, context_tokens, temperature=temp, topk=topk)
        words_per_bit = 1
    text = enc.decode(out)
    
    print(message)
    print(len(message))
    print("="*40 + " Encoding " + "="*40)
    print(text)
    print('ppl: %0.2f, kl: %0.3f, words/bit: %0.2f, bits/word: %0.2f, entropy: %.2f' % (math.exp(nll), kl, words_per_bit, 1/words_per_bit, Hq/0.69315))
    
    # Decode binary message from bits using the same arbitrary context
    if mode != 'sample':
        if mode == 'arithmetic':
            message_rec = decode_arithmetic(model, enc, text, context_tokens, temp=temp, precision=precision, topk=topk)
        elif mode == 'huffman':
            message_rec = decode_huffman(model, enc, text, context_tokens, block_size)
        elif mode == 'bins':
            message_rec = decode_block(model, enc, text, context_tokens, block_size, bin2words, words2bin)

        print("="*40 + " Recovered Message " + "="*40)
        print(message_rec)
        print("=" * 80)
        # Finally map message bits back to original text
        if unicode_enc:
            message_rec = [bool(item) for item in message_rec]
            ba = bitarray.bitarray(message_rec)
            reconst = ba.tobytes().decode('utf-8', 'ignore')
        else:
            reconst = encode_arithmetic(model, enc, message_rec, message_ctx, precision=40, topk=60000)
            reconst = enc.decode(reconst[0])
        print(reconst)

if __name__ == '__main__':
    main()
