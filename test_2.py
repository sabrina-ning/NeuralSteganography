import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoProcessor, Emu3ForCausalLM, Emu3ForConditionalGeneration

from transformers import AutoModel, AutoImageProcessor, AutoModelForCausalLM
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation import LogitsProcessorList, PrefixConstrainedLogitsProcessor, UnbatchedClassifierFreeGuidanceLogitsProcessor

from PIL import Image
import numpy as np
import time
import ipdb

from utils import is_cit, encode_image, decode_image, get_model, entropy, limit_past
from arithmetic import decode_arithmetic

import sys
sys.path.append('./Emu3')
from emu3.mllm.processing_emu3 import Emu3Processor

'''
Token sampling using original Emu3 source code models
'''

# model_id = "gpt2"
# model = AutoModelForCausalLM.from_pretrained(
#     model_id, 
#     torch_dtype=torch.float16, 
#     low_cpu_mem_usage=True,
#     device_map="cuda:0")
# enc = AutoTokenizer.from_pretrained(model_id)


# model_id = "BAAI/Emu3-Chat-hf"
# # model_id = "BAAI/Emu3-Gen-hf"
# model = Emu3ForConditionalGeneration.from_pretrained(
#     model_id, 
#     torch_dtype=torch.float16, 
#     low_cpu_mem_usage=True,
#     trust_remote_code=True,
#     device_map="cuda:0")
# base_model = model.model
# processor = AutoProcessor.from_pretrained(model_id)
# enc = processor.tokenizer


EMU_HUB = "BAAI/Emu3-Stage1"
# EMU_HUB = "BAAI/Emu3-Chat"
VQ_HUB = "BAAI/Emu3-VisionTokenizer"
model = AutoModelForCausalLM.from_pretrained(
    EMU_HUB,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    device_map="cuda:0")
tokenizer = AutoTokenizer.from_pretrained(
    EMU_HUB, 
    torch_dtype=torch.float16,
    trust_remote_code=True)
image_processor = AutoImageProcessor.from_pretrained(
    VQ_HUB, 
    torch_dtype=torch.float16,
    trust_remote_code=True)
image_tokenizer = AutoModel.from_pretrained(
    VQ_HUB, 
    torch_dtype=torch.float16,
    device_map="cuda:0", 
    trust_remote_code=True).eval()
processor = Emu3Processor(image_processor, image_tokenizer, tokenizer)
enc = processor.tokenizer


def test_prob_dist():
    print(type(model))
    print(type(enc))
    
    context_str = """Instagram is an American photo and short-form video sharing social networking service owned by Meta Platforms. It allows users to upload media that can be edited with filters, be organized by hashtags, and be associated with a location via geographical tagging. Posts can be shared publicly or with preapproved followers."""
    # context_str = """San Francisco, officially the City and County of San Francisco, is a commercial, financial, and cultural center of Northern California. With a population of 827,526 residents as of 2024, San Francisco is the fourth-most populous city in the U.S. state of California and the 17th-most populous in the United States."""

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

    # instruction = "Continue writing the following article. Feel free to add new paragraphs, newlines, and expand on the ideas presented."
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

    min_entropy = float('inf')
    max_entropy = -float('inf')

    breakpoint()

    with torch.no_grad():
        for i in range(1000):
            # if past:
            #     past = limit_past(past)
            out = model(input_ids=prev.unsqueeze(0), past_key_values=past, use_cache=True)
            logits = out.logits
            past = out.past_key_values

            # logits[0, -1, -1] = -1e4 # endoftext token can't happen
            logits[0, -1, 151850] = -1e4 # endofsequence can't happen
            # logits, indices = logits[0, -1, :].sort(descending=True)
            logits, indices = logits[0, -1, :151643].sort(descending=True)

            logits = logits.double()
            logits_temp = logits / temp
            probs_temp = F.softmax(logits_temp, dim=0)
            log_probs_temp = F.log_softmax(logits_temp, dim=0)
            
            selection = torch.multinomial(probs_temp[:topk], num_samples=1) # randomly sample index
            prev = indices[selection].view(1)
            output = torch.cat((output, prev))

            entropy_val = entropy(probs_temp, log_probs_temp)
            min_entropy = min(min_entropy, entropy_val)
            max_entropy = max(max_entropy, entropy_val)

            print("probs:", probs_temp[:10])
            print("entropy:", entropy_val)
            print("output:", enc.decode(output[len(context):].tolist()))
            print("generated:", i + 1)

            print()

            # time.sleep(0.1)
    
    print("min entropy:", min_entropy)
    print("max entropy:", max_entropy)

test_prob_dist()