import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoProcessor, Emu3ForCausalLM, Emu3ForConditionalGeneration
import time

from transformers import AutoModel, AutoImageProcessor, AutoModelForCausalLM
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation import LogitsProcessorList, PrefixConstrainedLogitsProcessor, UnbatchedClassifierFreeGuidanceLogitsProcessor

import secrets
from PIL import Image
import numpy as np
# from skimage.metrics import structural_similarity as ssim

from utils import is_cit, encode_image, decode_image, get_model, entropy
from arithmetic import decode_arithmetic

'''
Collection of tests to explore what different Emu3 model checkpoints do
'''

## ========== TEXT ==========

seed = 1234
temp = 0.9
top_k = 50
num_tokens = 1

context = "Cornell University is a private Ivy League research university based in Ithaca, New York, United States. The university was 173 years old in 2020 and is the second-oldest continuous family of universities in the United States. It is a member of the National Association of Independent"

def test_causal():
    model_name = "BAAI/Emu3-Chat"
    enc = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map="cuda"
    )
    model = Emu3ForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="cuda"
    )
    print(f"AutoModel: {type(model)}, AutoTokenizer: {type(enc)}")
    # torch.random.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    
    context_tokens = torch.tensor(enc.encode(context), device="cuda")
    output = []
    output_str = context

    with torch.no_grad():
        prev = context_tokens
        past = None
        
        # for i in range(num_tokens):
        while True:
            out = model(input_ids=prev.unsqueeze(0), past_key_values=past, use_cache=True)
            logits = out.logits
            past = out.past_key_values

            logits, indices = logits[0, -1, :].sort(descending=True)
            logits = logits.double()
            logits_temp = logits / temp
            probs_temp = F.softmax(logits_temp, dim=0)

            print(f"Top-{top_k} tokens:")
            for rank_idx in range(top_k):
                token_id = indices[rank_idx].item()
                token_text = enc.decode([token_id])
                print(f"  {rank_idx}: {token_text} ({token_id})")

            # prev = indices[torch.multinomial(probs_temp, num_samples=1)]
            i = int(input("Select index: "))
            if i == 100:
                break
            prev = indices[torch.tensor([i], device="cuda")]
            output.append(prev.item())
            output_str += enc.decode([prev.item()])

    return output, output_str

def test_conditional():
    model_name = "BAAI/Emu3-Chat-hf"
    enc = AutoTokenizer.from_pretrained(
        model_name,
        device_map="cuda"
    )
    model = Emu3ForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cuda"
    )
    print(f"AutoModel: {type(model)}, AutoTokenizer: {type(enc)}")
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    context_tokens = torch.tensor(enc.encode(context), device="cuda")
    output = []
    output_str = context

    with torch.no_grad():
        prev = context_tokens
        past = None

        for i in range(num_tokens):
            out = model(input_ids=prev.unsqueeze(0), past_key_values=past, use_cache=True)
            logits = out.logits
            past = out.past_key_values

            logits, indices = logits[0, -1, :151643].sort(descending=True)
            logits = logits.double()
            logits_temp = logits / temp
            probs_temp = F.softmax(logits_temp, dim=0)

            print(f"Top-{top_k} tokens:")
            for rank_idx in range(top_k):
                token_id = indices[rank_idx].item()
                token_text = enc.decode([token_id])
                print(f"  {rank_idx}: {token_text} ({token_id})")

            # prev = indices[torch.multinomial(probs_temp, num_samples=1)]
            prev = indices[torch.tensor([3], device="cuda")]
            output.append(prev.item())
            output_str += enc.decode([prev.item()])

    return output, output_str

# output_1, output_str_1 = test_causal()
# output_2, output_str_2 = test_conditional()
# print(output_1)
# print(" ", output_str_1)
# print(output_2)
# print(" ", output_str_2)
# print(output_1 == output_2)


def test_tokenizer():
    tokens = [16, 22, 18, 1635, 2310, 304, 220, 17, 15, 17, 15, 323, 374, 279, 2086, 12, 337, 4979, 19259, 2997, 315, 23106, 304, 279, 3639, 4180, 13, 1084, 374, 264, 4462, 315, 279, 5055, 10024, 315, 21994, 30383, 382, 785, 12103, 374, 279, 1156, 304, 279, 3639, 4180, 311, 3010, 264, 8381, 304, 28006, 15712, 13, 576, 6022, 315, 47704, 18637, 374, 14975, 438, 264, 4462, 315, 279, 10024, 315, 3693, 78119, 323, 74798, 323, 374, 21006, 4221, 279, 1909, 220, 17, 15, 28006, 8682, 304, 279, 3639, 4180, 304, 547, 808, 13, 5398, 609, 4337, 8259, 594, 220, 17, 15, 17, 16, 7107, 30383, 1140, 13, 1084, 374, 1083, 825]
    text = """173 years old in 2020 and is the second-oldest continuous family of universities in the United States. It is a member of the National Association of Independent Schools.

The university is the first in the United States to offer a degree in dental medicine. The School of Dental Medicine is recognized as a member of the Association of American Colleges and Universities and is ranked among the top 20 dental schools in the United States in U.S. News & World Report's 2021 Best Schools list. It is also one"""

    print(tokens)
    print()

    enc = AutoTokenizer.from_pretrained(
        "BAAI/Emu3-Chat",
        trust_remote_code=True,
        device_map="cuda:0")
    print(type(enc))
    # print(enc.decode(tokens))
    # print(enc.encode(enc.decode(tokens)))
    print(enc.encode(text))
    print(enc.decode(enc.encode(text)))
    print()
    
    enc = AutoTokenizer.from_pretrained(
        "BAAI/Emu3-Chat-hf",
        trust_remote_code=True,
        device_map="cuda:0")
    print(type(enc))
    # print(enc.decode(tokens))
    # print(enc.encode(enc.decode(tokens)))
    print(enc.encode(text))
    print(enc.decode(enc.encode(text)))
    print()
    
    enc = AutoProcessor.from_pretrained(
        "BAAI/Emu3-Chat-hf",
        trust_remote_code=True,
        device_map="cuda:0")
    print(type(enc))
    # print(enc.decode(tokens))
    # print(encode(enc, enc.decode(tokens)))
    # print(encode(enc, text))
    # print(enc.decode(encode(enc, text)))
    print()

    print(torch.tensor([enc.tokenizer.image_wrapper_token_id]).cuda())

# test_tokenizer()


def test_reversible():
    message = "endoftext<eos>"
    cover_tokens = [8691, 723, 427, 27, 84399, 29]

    tokenizer = AutoTokenizer.from_pretrained(
        "BAAI/Emu3-Stage1",
        trust_remote_code=True,
        device_map="cuda:0")
    print(type(tokenizer))

    # decoded = tokenizer.decode(cover_tokens)
    # print(decoded)
    # encoded = tokenizer.encode(decoded)
    # print(encoded)
    # print(cover_tokens == encoded)
    print(tokenizer.encode(message))
    print(tokenizer.decode(tokenizer.encode(message)))

    print()

    tokenizer = AutoTokenizer.from_pretrained(
        "BAAI/Emu3-Chat",
        trust_remote_code=True,
        device_map="cuda:0")
    print(type(tokenizer))

    # decoded = tokenizer.decode(cover_tokens)
    # print(decoded)
    # encoded = tokenizer.encode(decoded)
    # print(encoded)
    # print(cover_tokens == encoded)
    print(tokenizer.encode(message))
    print(tokenizer.decode(tokenizer.encode(message)))

    print()
    
    processor_hf = AutoProcessor.from_pretrained(
        "BAAI/Emu3-Chat-hf",
        trust_remote_code=True,
        device_map="cuda:0")
    tokenizer_hf = processor_hf.tokenizer
    print(type(tokenizer_hf))

    # decoded = tokenizer.decode(cover_tokens)
    # print(decoded)
    # encoded = tokenizer.encode(decoded)
    # print(encoded)
    # print(cover_tokens == encoded)
    print(tokenizer.encode(message))
    print(tokenizer.decode(tokenizer.encode(message)))

# test_reversible()


def random():
    seed = 12345
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    for i in range(10):
        dummy_list = [float(i) for i in range(2048)]
        print(torch.multinomial(torch.tensor(dummy_list, device="cuda"), num_samples=1))

    print()

    seed = 12345
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    for i in range(10):
        dummy_list = [float(i + 1) for i in range(2048)]
        print(torch.multinomial(torch.tensor(dummy_list, device="cuda"), num_samples=1))

    print()

    seed = 12345
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    for i in range(10):
        dummy_list = [float(2) for i in range(2048)]
        print(torch.multinomial(torch.tensor(dummy_list, device="cuda"), num_samples=1))

# random()


def test_text_models():
    model_name = "BAAI/Emu3-Chat"
    model = Emu3ForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="cuda"
    )
    print(type(model))
    print(type(model.model))
    print(model.config.is_encoder_decoder) # False; decoder-only model
    print(model.model.config.is_encoder_decoder)

    model_name = "BAAI/Emu3-Chat-hf"
    model = Emu3ForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="cuda"
    )
    print(type(model))
    print(type(model.model.text_model))
    print(model.config.is_encoder_decoder) # False

# test_text_models()


def find_first_difference(file1_path, file2_path):
    line_number = -1
    with open(file1_path, 'r', encoding='utf-8') as f1, open(file2_path, 'r', encoding='utf-8') as f2:
        for line_number, (line1, line2) in enumerate(zip(f1, f2), start=1):
            if line1 != line2:
                print(f"Difference found at line {line_number}:")
                print(f"{file1_path}: {line1.rstrip()}")
                print(f"{file2_path}: {line2.rstrip()}")
                return

        # Check for extra lines in either file
        extra_f1 = f1.readline()
        extra_f2 = f2.readline()
        if extra_f1:
            print(f"{file1_path} has extra lines starting at line {line_number + 1}:")
            print(extra_f1.rstrip())
        elif extra_f2:
            print(f"{file2_path} has extra lines starting at line {line_number + 1}:")
            print(extra_f2.rstrip())
        else:
            print("Files are identical.")

file1 = "output_encode.txt"
file2 = "output_decode.txt"
# find_first_difference(file1, file2)


def compare_tokenizers():
    cover_tokens = [1260, 7391, 806, 9664, 1635, 869, 2272, 11, 323, 8469, 389, 6652, 220, 17, 19, 11, 220, 16, 22, 24, 24, 382, 40, 614, 1730, 429, 1052, 525, 264, 5625, 315, 6032, 66145, 429, 1140, 6515, 594, 6168, 438, 264, 6277, 27994, 11, 438, 566, 15540, 3693, 8437, 1526, 279, 4116, 2348]

    enc = AutoTokenizer.from_pretrained("BAAI/Emu3-Chat-hf")
    print(type(enc))
    cover_text = enc.decode(cover_tokens)
    print(cover_text)
    print(enc.encode(cover_text))

    enc = AutoTokenizer.from_pretrained("gpt2")
    print(type(enc))
    cover_text = enc.decode(cover_tokens)
    print(cover_text)
    print(enc.encode(cover_text))

# compare_tokenizers()


def foo():
    enc = AutoTokenizer.from_pretrained("BAAI/Emu3-Chat-hf")
    # encoded = [55719, 3822, 702, 3635, 825, 315, 279, 1429, 40285, 14336, 315, 5080, 6832, 304, 279, 3639, 4180, 382, 641, 279, 220, 17, 15, 16, 23, 12, 17, 15, 16, 24, 14250, 1042, 11, 55719, 3822, 572, 21006, 671, 17, 18, 4221, 678, 584, 14336, 315, 5080, 6832, 304, 279, 3639, 4180, 553, 547, 808, 13, 5398, 609, 4337, 8259, 13, 55719, 374, 3881, 369, 1181, 46899, 14250, 7468, 11, 1181, 3746, 51021, 3922, 11, 323, 1181, 15155, 311, 14250, 37556, 13, 576, 12103, 6081, 916, 220, 18]
    # encoded = [1260, 14616, 304, 5163, 3080, 220, 16, 22, 24, 21, 11, 2337, 892, 882, 566, 6342, 264, 1376, 3476, 304, 279, 51031, 315, 12095, 11, 892, 9482, 279, 4116, 13, 4710, 6025, 9380, 5163, 11, 6515, 8469, 389, 6527, 220, 18, 11, 220, 16, 22, 24, 24, 11, 518, 279, 4231, 315, 220, 21, 21, 13, 5301, 4545, 572, 2270, 448, 23782, 37284, 323, 806, 19588, 438, 264, 7653]
    # decoded = enc.decode(encoded)
    # re_encoded = enc.encode(decoded) # same
    # print(encoded)
    # print(re_encoded)
    # decoded_2 = enc.decode(re_encoded)
    # re_encoded_2 = enc.encode(decoded_2) # same
    # print(re_encoded_2)

    # token = [715, 198]
    # print(token)
    # decoded = enc.decode(token)
    # print(f"[{decoded}]\n")
    # re_encoded = enc.encode(decoded)
    # print(re_encoded)

    # token = [382]
    # print(token)
    # decoded = enc.decode(token)
    # print(f"[{decoded}]\n")
    # re_encoded = enc.encode(decoded)
    # print(re_encoded)

    # token = [4292, 37]
    # print(token)
    # decoded = enc.decode(token)
    decoded = "\n\n"
    print(f"START: [{decoded}]\n")
    re_encoded = enc.encode(decoded)
    print(re_encoded)
    re_decoded = enc.decode(re_encoded)
    print(re_decoded)

    decoded = "\n\nA"
    print(f"START: [{decoded}]\n")
    re_encoded = enc.encode(decoded)
    print(re_encoded)
    re_decoded = enc.decode(re_encoded)
    print(re_decoded)
    
    # 13:   .
    # 271:  ĊĊ
    # 4710: ĠĊĊ
    # 6025: After
    # 1406: ĊĊĊ
    # 1022: ĊĊĊĊ

    # [13, 4710, 6025]  -> [13, 715, 198, 6025]
    # [    4710, 6025]  -> [    715, 198, 6025]
    # [13, 4710]        -> [13, 4710]
    # [    4710]        -> [    4710]

    # [13, 271, 6025]   -> [13, 198, 198, 6025]
    # [    271, 6025]   -> [    198, 198, 6025]
    # [13, 271]         -> [13, 271]
    # [    271]         -> [    271]
    # [13, 271, 198]    -> [13, 1406]
    # [13, 271, 198, 198] -> [13, 1022]

    # [1022]            -> [1022,      13]
    # [1022, 13]        -> [1406, 198, 13]
    # [1406, 1022]      -> [34583]
    
    # [198, 1406]       -> [1022]
    # [1406, 198]       -> [1022]
    
# foo()


def print_decode_encode(enc, tokens):
    print(tokens)
    decoded = enc.decode(tokens)
    # print(f"[{decoded}]")

    tokens_str_list = []
    for token in tokens:
        tokens_str_list.append(enc.decode([token]))
    print(tokens_str_list, "\n")

    print(decoded, "\n")

    re_encoded = enc.encode(decoded)
    print(re_encoded)
    
    re_encoded_str_list = []
    for token in re_encoded:
        re_encoded_str_list.append(enc.decode([token]))
    print(re_encoded_str_list)

    print(enc.decode(re_encoded))

    print("="*20)

# enc = AutoTokenizer.from_pretrained("BAAI/Emu3-Chat-hf", trust_remote_code=True)

# print_decode_encode(enc, [271, 32]) # 715, 198, 6025
# print_decode_encode(enc, [271])
# print_decode_encode(enc, [4710, 38131]) # 715, 198, 38131
# print_decode_encode(enc, [7806, 9070]) # 12, 23169
# print_decode_encode(enc, [4686, 2220, 580, 14728]) # 12, 983, 12, 85168
# print_decode_encode(enc, [4292]) # 568, 198
# print_decode_encode(enc, [382, 641]) # 13, 198, 198, 641
# print_decode_encode(enc, [22092]) # 12, 10473

# print_decode_encode(enc, [95705]) # 12, 68, 2209

# print_decode_encode(enc, [4686, 2220, 580, 14728])
# print_decode_encode(enc, [1744, 388])
# print_decode_encode(enc, [1744, 32])
# print_decode_encode(enc, [68022])

# print_decode_encode(enc, [271])
# print_decode_encode(enc, [271, 198])
# print_decode_encode(enc, [271, 32])

# print_decode_encode(enc, [520, 1512, 613, 12586, 42516, 25682, 10171, 9, 330, 11654, 329, 37205, 3263, 55719, 3822, 13, 366, 2428, 1110, 2136, 520, 1512, 613, 12586, 22258, 38030, 8546, 14, 49857, 37205, 10171, 9, 330, 33768, 59363, 3263, 55719, 3822, 13, 366, 2428, 1110, 2136, 520, 1512, 613, 12586, 92477, 96102, 3448, 10171, 9, 330, 94667, 323, 16951, 3263, 55719, 3822, 13, 366, 2428, 1110, 2136, 520, 1512, 613, 12586, 6663, 580, 14728, 9777, 5477, 2649, 9685, 715, 9, 330, 7916, 287, 5633, 20173, 3263, 55719, 3822, 13, 366, 2428, 1110, 2136, 520, 1512, 613, 12586, 60370, 287, 413, 37218, 9685, 715, 9, 330, 2101, 56865, 3263, 55719, 3822, 13, 366, 2428, 1110, 2136, 520, 1512, 613, 12586, 52120, 56865, 9685, 715, 9, 330, 13424, 3263, 55719, 3822, 13, 366, 2428, 1110, 2136, 520, 1512, 613, 12586, 7530, 95698])
# print_decode_encode(enc, [13, 38630, 613, 13, 55101, 14, 9096, 12, 355, 9685, 198, 9, 330, 11654, 329, 37205, 3263, 55719, 3822, 13, 366, 2428, 1110, 2136, 13, 38630, 613, 13, 55101, 14, 12779, 12, 370, 8546, 14, 49857, 37205, 9685, 198, 9, 330, 33768, 59363, 3263, 55719, 3822, 13, 366, 2428, 1110, 2136, 13, 38630, 613, 13, 55101, 14, 28402, 12, 826, 3448, 9685, 198, 9, 330, 94667, 323, 16951, 3263, 55719, 3822, 13, 366, 2428, 1110, 2136, 13, 38630, 613, 13, 55101, 14, 85168, 12, 437, 12, 27122, 9685, 220, 198, 9, 330, 7916, 287, 5633, 20173, 3263, 55719, 3822, 13, 366, 2428, 1110, 2136, 13, 38630, 613, 13, 55101, 14, 23362, 287, 413, 37218, 9685, 220, 198, 9, 330, 2101, 56865, 3263, 55719, 3822, 13, 366, 2428, 1110, 2136, 13, 38630, 613, 13, 55101, 14, 278, 56865, 9685, 220, 198, 9, 330, 13424, 3263, 55719, 3822, 13, 366, 2428, 1110, 2136, 13, 38630, 613, 13, 55101, 14, 43441, 938])


def test_decode_only():
    model = Emu3ForConditionalGeneration.from_pretrained("BAAI/Emu3-Chat-hf",
        torch_dtype=torch.float16,
        device_map="cuda"
    )
    # enc = AutoTokenizer.from_pretrained("BAAI/Emu3-Chat-hf")

    context_tokens = [151643, 90541, 613, 3822, 374, 264, 869, 56879, 8953, 3412, 12103, 3118, 304, 358, 339, 17106, 11, 1532, 4261, 11, 3639, 4180, 13, 576, 12103, 572, 220, 16, 17, 339, 304, 3793, 315, 38048, 518, 279, 220, 17, 15, 17, 15, 14, 17, 16, 14250, 1042, 13, 576, 1482, 5458, 2487, 702, 264]
    text = " student-to-faculty ratio"
    # decode_arithmetic(model, enc, text, context_tokens, temp=0.9, precision=26, topk=300)

# test_decode_only()


def test_is_cit():
    # print(is_cit(enc, 32, []))
    # print(is_cit(enc, 32, [4710]))
    # print(is_cit(enc, 32, [32]))
    # print(is_cit(enc, 4710, []))
    return

# test_is_cit()


# Generates a cryptographically secure random string of 0's and 1's
def generate_random_bits():
    message_len = 1800
    num_messages = 2

    for _ in range(num_messages):
        res = []
        for _ in range(message_len):
            res.append(secrets.choice([0, 1]))
        print(res)
        print()

# generate_random_bits()


def check_prefix_tokenization():
    enc = AutoTokenizer.from_pretrained("BAAI/Emu3-Chat", trust_remote_code=True)

    count = 0
    for i in range(151642):
        if [i] != enc.encode(enc.decode([i])):
            count += 1
            print("false")
    print(count)

# check_prefix_tokenization()


## ========== IMAGES ==========

# model_id = "gpt2"
model_id = "BAAI/Emu3-Gen-hf"
model = Emu3ForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    device_map="cuda:0")
base_model = model.model

processor = AutoProcessor.from_pretrained(model_id, torch_dtype=torch.float16)
enc = processor

# torch.set_printoptions(profile="full")

def encode_decode_image(image_path):
    downsample_ratio = 8

    # Encode: pixel values -> visual tokens

    image = Image.open(image_path).convert("RGB") # returns Image object
    width, height = image.size
    print(image.size)

    image = processor.image_processor.preprocess(image, return_tensors="pt")
    pixel_values = image["pixel_values"].to(torch.float16).cuda()
    print(pixel_values.shape) # [batches, channels, height', width']
    image_sizes = image["image_sizes"].cuda()
    print(image_sizes) # [[height', width']]

    eof_token_id = processor.tokenizer.eof_token_id
    eoi_token_id = processor.tokenizer.eoi_token_id
    eos_token_id = processor.tokenizer.eos_token_id

    with torch.no_grad():
        image_tokens = base_model.get_image_tokens(pixel_values, image_sizes).cuda()
        print(image_tokens.shape) # [height' * (width' + 1)]
        
        end_tokens = torch.tensor([eof_token_id, eoi_token_id, eos_token_id], device="cuda")
        image_tokens = torch.cat([image_tokens, end_tokens])
        print(image_tokens.shape) # [height' * (width' + 1) + 3]

    # Decode: visual tokens -> pixel values
    image_new = base_model.decode_image_tokens(image_tokens.unsqueeze(0), height=(height // downsample_ratio), width=(width // downsample_ratio))
    image_new = processor.image_processor.postprocess(image_new, return_tensors="PIL.Image.Image")['pixel_values'][0]
    image_new.save("new_images/image_small_decoded.jpg")

enc.image_processor.min_pixels = 128 * 128
encode_decode_image("new_images/touchdown_small.jpg")
# encode_decode_image("images/fruits.jpg")
# encode_decode_image("images/cat.jpg")
# encode_decode_image("images/cornell.jpg")


def test_encode_image(image_path):
    image_tokens = encode_image(image_path, model, enc.image_processor)
    print(image_tokens)

# test_encode_image("cat_64.jpg")


# print(enc.image_processor)
'''
Emu3ImageProcessor {
  "auto_map": {
    "AutoImageProcessor": "BAAI/Emu3-VisionTokenizer--image_processing_emu3visionvq.Emu3VisionVQImageProcessor"
  },
  "do_convert_rgb": true,
  "do_normalize": true,
  "do_rescale": true,
  "do_resize": true,
  "image_mean": [
    0.5,
    0.5,
    0.5
  ],
  "image_processor_type": "Emu3ImageProcessor",
  "image_std": [
    0.5,
    0.5,
    0.5
  ],
  "max_pixels": 1048576, (1024*1024)
  "min_pixels": 262144, (512*512)
  "processor_class": "Emu3Processor",
  "resample": 3,
  "rescale_factor": 0.00392156862745098,
  "size": {
    "max_pixels": 1048576,
    "min_pixels": 262144
  },
  "spatial_factor": 8
}
'''

# <|extra_203|> (151849)
# print(enc.tokenizer.bos_token_id)

def test_recovered_image(image_path):
    image_tokens, height, width, image_PIL = encode_image(image_path, model, enc)

    print(image_tokens)
    
    image = decode_image(image_tokens[1:], height, width, model, enc)
    # image = image.resize((512, 512))
    image.save(image_path + ".jpg")

    # calculate similary of images
    # original = np.array(image_PIL)
    # print(original.shape)
    # recovered = np.array(image)
    # print(recovered.shape)

    # ssim_score = ssim(original, recovered, multichannel=True, data_range=255, channel_axis=2)
    # print(ssim_score)


enc.image_processor.min_pixels = 128 * 128
# print(enc.image_processor)
# test_recovered_image("example_images/cornell_64.jpg")


def foo(variable=10):
    for i in range(5):
        variable -= 1
        print(variable)
    print(variable)

# foo()

# get_model(model_name="BAAI/Emu3-Chat-hf")


def test_prob_dist():

    print(type(model))
    print(type(enc))
    
    context_str = """Instagram is an American photo and short-form video sharing social networking service owned by Meta Platforms. It allows users to upload media that can be edited with filters, be organized by hashtags, and be associated with a location via geographical tagging. Posts can be shared publicly or with preapproved followers."""
    context = torch.tensor(enc.tokenizer.encode(context_str), device="cuda")
    prev = context
    output = context
    past = None

    num_tokens = 0

    with torch.no_grad():
        
        for i in range(1000):
            out = model(input_ids=prev.unsqueeze(0), past_key_values=past, use_cache=True)
            logits = out.logits
            past = out.past_key_values

            logits, indices = logits[0, -1, :151643].sort(descending=True)

            logits = logits.double()
            logits_temp = logits / temp
            probs_temp = F.softmax(logits_temp, dim=0)
            log_probs_temp = F.log_softmax(logits_temp, dim=0)
            
            selection = torch.multinomial(probs_temp[:300], num_samples=1) # FIXME
            prev = indices[selection].view(1)
            output = torch.cat((output, prev))

            num_tokens += 1

            print("probs:", probs_temp[:10])
            print("entropy:", entropy(probs_temp, log_probs_temp))
            print("output:", enc.tokenizer.decode(output[len(context):].tolist()))
            print("generated:", num_tokens)

            print()

            # time.sleep(1)

# test_prob_dist()