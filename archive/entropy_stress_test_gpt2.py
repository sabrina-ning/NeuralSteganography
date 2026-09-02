import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoProcessor, Emu3ForCausalLM, Emu3ForConditionalGeneration, AutoModel, AutoImageProcessor, AutoModelForCausalLM
from transformers import DynamicCache

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import time
import ipdb

from utils import is_cit, encode_image, decode_image, get_model, entropy, limit_past
from arithmetic import decode_arithmetic

'''
Entropy-sampling stress-tests using GPT2, similar to Emu3 experiments in test_3.py
'''

# seed = 1234
# np.random.seed(seed)
# torch.random.manual_seed(seed)
# torch.cuda.manual_seed(seed)

model_id = "openai-community/gpt2-xl"
enc = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="cuda"
)

print('model loaded!')

# returns exponential moving average, where alpha is smoothing factor
# def update_ema(prev_ema, new_val, alpha=0.01):
#     if not prev_ema:
#         return new_val
#     return alpha * new_val + (1 - alpha) * prev_ema

def crop_last(past, max_length: int):
    """Crop the past key values up to a new `max_length` in terms of tokens. `max_length` must be non-negative."""
    if past.get_seq_length() <= max_length:
        return
    past._seen_tokens = max_length
    for idx in range(len(past.key_cache)):
        if past.key_cache[idx].numel():
            past.key_cache[idx] = past.key_cache[idx][:, :, -max_length:, :]
            # breakpoint()
            past.value_cache[idx] = past.value_cache[idx][:, :, -max_length:, :]
            # breakpoint()
    return past

def get_cache_length(past):
    if isinstance(past[0], tuple):
        return past[0][0].shape[2]  # shape is [batch, heads, seq_len, dim]
    return past[0].shape[2]

def test_prob_dist(test_num):
    print(type(model))
    print(type(enc))
    
    context_str = """San Francisco, officially the City and County of San Francisco, is a commercial, financial, and cultural center of Northern California. With a population of 827,526 residents as of 2024, San Francisco is the fourth-most populous city in the U.S. state of California and the 17th-most populous in the United States."""

    context = torch.tensor(enc.encode(context_str), device="cuda")

    prev = context # input to forward pass, last token generated
    output = torch.tensor([], device="cuda", dtype=torch.long)

    past = None # cached key-value pairs

    temp = 0.9
    topk = 300

    # ema_entropy = None
    # lower_thresh = 0.1
    # upper_thresh = 5.0

    acc_entropy = 0
    acc_values = []
    entropy_values = []

    breakpoint()
    truncate_len = 200
    tokens_since_truncate = 0

    for i in range(2000):
        # if tokens_since_truncate >= truncate_len: # this means cache is of size context_len
        #     past = limit_past(past, max_len=truncate_len)
        #     tokens_since_truncate = 0
        #     absolute_pos = truncate_len - 1 # keep [0..199], so new would be [200]
        #     breakpoint()

        # with torch.no_grad():
        #     # out = model(input_ids=prev.unsqueeze(0), past_key_values=past, use_cache=True)
        #     if past is None:
        #         # breakpoint()
        #         out = model(input_ids=prev.unsqueeze(0), use_cache=True)
        #         absolute_pos = len(context)
        #         # breakpoint()
        #     else:
        #         # breakpoint()
        #         print('D')
        #         # cache_position should be position where input_ids will be written in the cache
        #         cache_position = torch.tensor([absolute_pos], device="cuda")

        #         out = model(input_ids=prev.unsqueeze(0), 
        #                     cache_position=cache_position, 
        #                     position_ids=cache_position.unsqueeze(0), 
        #                     past_key_values=past, 
        #                     use_cache=True)
        #         absolute_pos += 1
                # breakpoint()

        if tokens_since_truncate >= truncate_len and past is not None:
            past = limit_past(past, max_len=truncate_len)
            tokens_since_truncate = 0
            breakpoint()

        with torch.no_grad():
            if past is None:
                out = model(input_ids=prev.unsqueeze(0), use_cache=True)
            else:
                # cache_position = where the new token will be written
                cache_position = torch.tensor([get_cache_length(past)], device="cuda")
                position_ids = torch.tensor([[len(context) + len(output)]], device="cuda")
                
                out = model(input_ids=prev.unsqueeze(0), 
                            cache_position=cache_position, 
                            position_ids=position_ids, 
                            past_key_values=past, 
                            use_cache=True)

        logits = out.logits
        past = out.past_key_values # update cache
        print(type(past))

        # print(len(past))
        # past = limit_past(past)
        # print(len(past))

        # logits[0, -1, enc.eos_token_id] = -1e4 # endoftext token can't happen
        # logits[0, -1, 151850] = -1e4 # endofsequence can't happen
        # logits, indices = logits[0, -1, :].sort(descending=True)

        # limit to only text tokens in output
        logits, indices = logits[0, -1, :-1].sort(descending=True)

        # next_token_logits = logits[0, -1, :]
        # sorted_logits, indices = next_token_logits.sort(descending=True)

        # logits_temp = sorted_logits.double() / temp
        # probs_temp = F.softmax(logits_temp, dim=-1)
        # log_probs_temp = F.log_softmax(logits_temp, dim=-1)
        
        # selection = torch.multinomial(probs_temp[:topk], num_samples=1) # randomly sample index
        # prev = indices[selection].view(1)
        # output = torch.cat((output, prev))

        logits = logits.double()
        logits_temp = logits / temp
        probs_temp = F.softmax(logits_temp, dim=0)
        log_probs_temp = F.log_softmax(logits_temp, dim=0)
        
        selection = torch.multinomial(probs_temp[:topk], num_samples=1) # randomly sample index

        print(f"Current token ID: {selection}, Vocab size: {len(indices)}")
        if selection < 0 or selection >= len(indices):
            print("!!! INVALID TOKEN ID DETECTED !!!")

        prev = indices[selection].view(1)
        output = torch.cat((output, prev))
        tokens_since_truncate += 1

        entropy_val = entropy(probs_temp, log_probs_temp)
        # ema_entropy = update_ema(ema_entropy, entropy_val)
        acc_entropy += entropy_val
        acc_values.append(acc_entropy)
        entropy_values.append(entropy_val)

        # FIXME adjust temperature
        # if ema_entropy < lower_thresh:
        #     temp *= 1.1
        # elif ema_entropy > upper_thresh:
        #     temp /= 1.1
        # temp = max(0.5, min(temp, 2.0))

        print("probs:", probs_temp[:10])
        print("entropy:", entropy_val)
        print("output:", enc.decode(output.tolist()))
        print("generated:", i + 1)

        print()

    # plt.plot(acc_values, marker='o')
    # plt.xlabel("Iteration")
    # plt.ylabel("Accumulated Value")
    # plt.savefig("acc_entropy_" + str(test_num) + ".png")

    # fig, (ax1, ax2) = plt.subplots(2, 1)

    # ax1.plot(values, marker='o')
    # ax1.set_xlabel("Iteration")
    # ax1.set_ylabel("Accumulated Entropy")

    # ax2.plot(entropy_values, marker='o')
    # ax2.set_xlabel("Iteration")
    # ax2.set_ylabel("Entropy")

    # plt.savefig("entropy_" + str(test_num) + ".png")

for test_num in range(5):
    print('C')
    test_prob_dist(test_num)
