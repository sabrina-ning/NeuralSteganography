import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoProcessor,
    Emu3ForCausalLM,
    Emu3ForConditionalGeneration,
    AutoModel,
    AutoImageProcessor,
    AutoModelForCausalLM,
    DynamicCache,
)
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time
import ipdb

from utils import is_cit, encode_image, decode_image, get_model, entropy, limit_past
from arithmetic import decode_arithmetic

'''
Repeats GPT2 experiment in test_4.py, but explores different cache truncation logic
'''

seed = 12345
np.random.seed(seed)
torch.random.manual_seed(seed)
torch.cuda.manual_seed(seed)

print("start")

model_id = "openai-community/gpt2-xl"
# model_id = "Qwen/Qwen3-0.6B"
enc = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="cuda",
)


def test_prob_dist(test_num):
    print(type(model))
    print(type(enc))

    context_str = """San Francisco, officially the City and County of San Francisco, is a commercial, financial, and cultural center of Northern California. With a population of 827,526 residents as of 2024, San Francisco is the fourth-most populous city in the U.S. state of California and the 17th-most populous in the United States."""
    context = torch.tensor(enc.encode(context_str), device="cuda")

    prev = context
    output = context
    past = None

    temp = 0.9
    topk = 100

    breakpoint()

    # keep_length = 512
    # max_length = 1022

    for i in range(1000):
        if past is not None and i % 100 == 0:
            print(f"truncating at token {i}, original prev length {len(prev)}")
            # breakpoint()
            past = None
            prev = output[-100:]
            # breakpoint()

        with torch.no_grad():
            out = model(
                input_ids=prev.unsqueeze(0),
                past_key_values=past,
                use_cache=True,
            )
            # out = model(input_ids=output.unsqueeze(0), use_cache=False)

        logits = out.logits
        past = DynamicCache.from_legacy_cache(out.past_key_values)  # update cache
        # past = out.past_key_values

        print(
            "cache info:",
            past._seen_tokens,
            len(past.key_cache),
            len(past.value_cache),
            "\n",
        )

        # limit to only text tokens in output
        logits, indices = logits[0, -1, :-1].sort(descending=True)

        logits = logits.double()
        logits_temp = logits / temp
        probs_temp = F.softmax(logits_temp, dim=0)
        log_probs_temp = F.log_softmax(logits_temp, dim=0)

        selection = torch.multinomial(probs_temp[:topk], num_samples=1)  # sample index

        print(f"Current token ID: {selection}, Vocab size: {len(indices)}")
        if selection < 0 or selection >= len(indices):
            print("INVALID TOKEN ID")

        prev = indices[selection].view(1)
        output = torch.cat((output, prev))
        # output = torch.cat((output, indices[selection].view(1)))

        print("probs:", probs_temp[:10])
        print("output:", enc.decode(output[len(context):].tolist()))
        print("generated:", i + 1)
        print()


for test_num in range(5):
    print(f"test {test_num}")
    test_prob_dist(test_num)
