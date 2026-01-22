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
Stress-tests text generation (entropy dynamics) on Emu3 model, explores cache truncation and plots entropy trace
'''

# model_id = "gpt2"
# model = AutoModelForCausalLM.from_pretrained(
#     model_id, 
#     torch_dtype=torch.float16,
#     device_map="cuda:0")
# enc = AutoTokenizer.from_pretrained(model_id)


model_id = "BAAI/Emu3-Chat-hf"
# model_id = "BAAI/Emu3-Gen-hf"
model = Emu3ForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.float16,
    device_map="cuda"
)
base_model = model.model
processor = AutoProcessor.from_pretrained(model_id)
enc = processor.tokenizer

# returns exponential moving average, where alpha is smoothing factor
def update_ema(prev_ema, new_val, alpha=0.01):
    if not prev_ema:
        return new_val
    return alpha * new_val + (1 - alpha) * prev_ema

def test_prob_dist(test_num):
    print(type(model))
    print(type(enc))
    
    # context_str = """Instagram is an American photo and short-form video sharing social networking service owned by Meta Platforms. It allows users to upload media that can be edited with filters, be organized by hashtags, and be associated with a location via geographical tagging. Posts can be shared publicly or with preapproved followers."""
    context_str = """San Francisco, officially the City and County of San Francisco, is a commercial, financial, and cultural center of Northern California. With a population of 827,526 residents as of 2024, San Francisco is the fourth-most populous city in the U.S. state of California and the 17th-most populous in the United States."""

    context = torch.tensor(enc.encode(context_str), device="cuda")
    prev = context
    output = context

    # inputs = processor(
    #     text=[context_str],
    #     mode='G', # G for generation, U for understanding
    #     return_tensors="pt"
    # ).to(model.device)
    # context = inputs["input_ids"].squeeze(0)
    # prev = context
    # output = context

    # instruction = "Continue writing the following context. Add new lines and new paragraphs as needed to expand on the ideas presented."
    # full_context = f"{instruction}\n\n{context_str}"
    # conversation = [
    #     {
    #         "role": "user",
    #         "content": [
    #             {
    #                 "type": "text",
    #                 "text": full_context
    #             }
    #         ],
    #     }
    # ]
    # inputs = processor.apply_chat_template(
    #     conversation,
    #     add_generation_prompt=True,
    #     tokenize=True,
    #     return_dict=True,
    #     return_tensors="pt"
    # ).to("cuda:0", dtype=torch.float16)
    # context = inputs["input_ids"].squeeze(0)
    # prev = context
    # output = context
    # print("context:", context)
    
    past = None

    temp = 0.9
    topk = 300

    # FIXME adjust these params
    # ema_entropy = None
    # lower_thresh = 1.0
    # upper_thresh = 7.0

    acc_entropy = 0
    acc_values = []
    entropy_values = []

    # breakpoint()

    with torch.no_grad():
        for i in range(10000):
            # if past:
            #     past = limit_past(past)

            if i != 0 and i % 1000 == 0:
                # temp = past[0][0].shape
                past = limit_past(past)
                # print(str(temp) + " -> " + str(past[0][0].shape))
                # breakpoint()

            out = model(input_ids=prev.unsqueeze(0), past_key_values=DynamicCache.from_legacy_cache(past), use_cache=True)
            logits = out.logits
            past = out.past_key_values

            # logits[0, -1, -1] = -1e4 # endoftext token can't happen
            # logits[0, -1, 151850] = -1e4 # endofsequence can't happen
            # logits, indices = logits[0, -1, :].sort(descending=True)

            # limit to only text tokens in output
            logits, indices = logits[0, -1, :151643].sort(descending=True)

            logits = logits.double()
            logits_temp = logits / temp
            probs_temp = F.softmax(logits_temp, dim=0)
            log_probs_temp = F.log_softmax(logits_temp, dim=0)
            
            selection = torch.multinomial(probs_temp[:topk], num_samples=1) # randomly sample index
            prev = indices[selection].view(1)
            output = torch.cat((output, prev))

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
            print("output:", enc.decode(output[len(context):].tolist()))
            print("generated:", i + 1)

            print()

    plt.plot(acc_values, marker='o')
    plt.xlabel("Iteration")
    plt.ylabel("Accumulated Value")
    plt.savefig("acc_entropy_" + str(test_num) + ".png")

    # fig, (ax1, ax2) = plt.subplots(2, 1)

    # ax1.plot(values, marker='o')
    # ax1.set_xlabel("Iteration")
    # ax1.set_ylabel("Accumulated Entropy")

    # ax2.plot(entropy_values, marker='o')
    # ax2.set_xlabel("Iteration")
    # ax2.set_ylabel("Entropy")

    # plt.savefig("entropy_" + str(test_num) + ".png")

for test_num in range(5):
    test_prob_dist(test_num)
