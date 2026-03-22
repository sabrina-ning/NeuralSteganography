import torch
import torch.nn.functional as F
from transformers import DynamicCache
import random
import time
# import numpy as np
# import matplotlib.pyplot as plt

from utils import limit_past, kl, entropy, bits2int, int2bits, is_sent_finish, num_same_from_beg, is_cit

def get_gen_mask(input_ids, logits, width, height):
    """
    Returns a mask to constrain image generation (zeros for valid, -inf for invalid)
    """
    # special tokens
    image_wrapper_token_id = 151851
    eol_token_id = 151846
    eof_token_id = 151847
    eoi_token_id = 151853
    eos_token_id = 151850
    pad_token_id = 151643

    visual_start_id = 151854

    mask = torch.full_like(logits, -float('inf'), device="cuda")

    # find start of image
    image_wrappers = (input_ids == image_wrapper_token_id).nonzero(as_tuple=True)
    if len(image_wrappers[1]) == 0:
        # if no image wrapper token, force start of image
        mask[image_wrapper_token_id] = 0.0
        return mask, True # deterministic
        
    # calculate offset from last image wrapper
    offset = input_ids.shape[1] - image_wrappers[1][-1]
    print("input ids shape:", input_ids.shape)

    row_length = width + 1
    is_deterministic = True
    
    if offset % row_length == 0: # finished row of visual tokens -> end of line
        mask[eol_token_id] = 0.0
    elif offset == row_length * height + 1: # finished generating all image tokens -> end of frame
        mask[eof_token_id] = 0.0
    elif offset == row_length * height + 2:
        mask[eoi_token_id] = 0.0
    elif offset == row_length * height + 3:
        mask[eos_token_id] = 0.0
    elif offset > row_length * height + 3: # if forced to continue generating tokens
        # mask[pad_token_id] = 0.0
        mask[image_wrapper_token_id] = 0.0
    else: # continue visual tokens
        mask[visual_start_id:] = 0.0
        is_deterministic = False
        
    return mask, is_deterministic

# === ENCODER (Bits -> Tokens) ===    
# message is a list of bits
def encode_arithmetic(model, enc, message, context, width=90, height=90, finish_sent=False, device='cuda', temp=1.0, precision=16, topk=None):
    start_time = time.perf_counter()
    # print("message:", message)
    total_num_bits = len(message)
    
    if isinstance(context, list):
        context = torch.tensor(context, device=device, dtype=torch.long)

    # Correct EOS ID for stopping condition
    eos_token_id = 151850

    max_val = 2**precision
    cur_interval = [0, max_val] 

    prev = context
    output = context
    past = None
    acc_output = torch.tensor([], device=device, dtype=torch.long)
    
    # Stats
    total_log_probs = 0
    total_kl = 0
    num_bits = 0
    total_num_for_stats = 0
    
    with torch.no_grad():
        i = 0
        
        while True:
            out = model(input_ids=prev.unsqueeze(0), past_key_values=past, use_cache=True)
            logits = out.logits
            past = out.past_key_values
            
            next_token_logits = logits[0, -1, :].to(dtype=torch.float32)
            
            mask, is_deterministic = get_gen_mask(output.unsqueeze(0), next_token_logits, width, height)
            masked_logits = next_token_logits + mask

            # If deterministic (EOL/EOF/EOI/EOS), select without consuming bits
            if is_deterministic:
                selection = torch.argmax(masked_logits).item()
                prev = torch.tensor([selection], device=device)
                output = torch.cat((output, prev))
                
                # Stop if we hit the EOS token
                if selection == eos_token_id:
                    # update accumulated output
                    acc_output = torch.cat((acc_output, output))

                    if num_bits >= total_num_bits:
                        print(f"\nAll {num_bits} bits encoded")
                        breakpoint()
                        break
                    else:
                        print(f"\nImage complete, but ({num_bits}/{total_num_bits}) bits remain. Starting new image...")
                        prev = context
                        output = context
                        past = None
                continue # skip rest of logic if deterministic

            logits_temp = masked_logits / temp

            # indices_to_remove = logits_temp < torch.topk(logits_temp, topk)[0][..., -1, None] # remove if prob less than last token of top-k
            # next_token_logits = logits_temp.masked_fill(indices_to_remove, -float('inf'))

            # probs = F.softmax(next_token_logits, dim=-1)
            # prev = torch.multinomial(probs, num_samples=1)
            # output = torch.cat((output, prev), dim=0)

            probs_temp = F.softmax(logits_temp, dim=0)

            probs_sorted, indices_sorted = torch.sort(probs_temp, descending=True)
            print("probs:", probs_sorted[:10])

            cur_int_range = cur_interval[1] - cur_interval[0]
            cur_threshold = 1.0 / cur_int_range
            
            cutoff_indices = (probs_sorted < cur_threshold).nonzero()
            if len(cutoff_indices) > 0:
                k = max(2, cutoff_indices[0].item())
            else:
                k = len(probs_sorted)
            
            if topk:
                k = min(k, topk)
        
            # # Cutoff / Top-K Logic
            # cur_int_range = cur_interval[1] - cur_interval[0]
            # cur_threshold = 1.0 / cur_int_range
            
            # cutoff_indices = (probs_temp < cur_threshold).nonzero()
            # if len(cutoff_indices) > 0:
            #     k = max(2, cutoff_indices[0].item())
            # else:
            #     k = len(probs_temp)
            
            # if topk:
            #     k = min(k, topk)
                
            # # Sort
            # probs_sorted, indices_sorted = torch.sort(probs_temp, descending=True)
            # print("probs:", probs_sorted[:10])
            
            probs_temp_int = probs_sorted[:k]
            indices = indices_sorted[:k]

            # Re-normalize and Round
            probs_temp_int = probs_temp_int / probs_temp_int.sum() * cur_int_range
            probs_temp_int = probs_temp_int.round().long()
            cum_probs = probs_temp_int.cumsum(0)
            
            # Fix Overfill
            overfill_index = (cum_probs > cur_int_range).nonzero()
            if len(overfill_index) > 0:
                cum_probs = cum_probs[:overfill_index[0]]
                indices = indices[:len(cum_probs)]
            
            cum_probs[-1] += cur_int_range - cum_probs[-1]
            
            # AC Selection
            probs_final = cum_probs.clone()
            print("k:", k)
            print("probs final:", probs_final[:10])
            probs_final[1:] = cum_probs[1:] - cum_probs[:-1]
            cum_probs += cur_interval[0]

            # Read bits
            # message_bits = message[i:i+precision]
            if i + precision > len(message):
                # message_bits = message_bits + [0] * (i + precision - len(message))
                # message_bits = message_bits + padding

                # Pad with random bits instead of zeros to maintain image entropy
                padding = [random.randint(0, 1) for _ in range(i + precision - len(message))] 
                message.extend(padding)

            message_bits = message[i : i + precision]
            message_idx = bits2int(reversed(message_bits))
            
            selection_idx = (cum_probs > message_idx).nonzero()[0].item()
            # selection = indices[selection_idx].item()
            
            # Update Interval
            new_int_bottom = cum_probs[selection_idx-1] if selection_idx > 0 else cur_interval[0]
            new_int_top = cum_probs[selection_idx]

            new_int_bottom_bits_inc = list(reversed(int2bits(new_int_bottom, precision)))
            new_int_top_bits_inc = list(reversed(int2bits(new_int_top-1, precision)))
            
            num_bits_encoded = num_same_from_beg(new_int_bottom_bits_inc, new_int_top_bits_inc)
            i += num_bits_encoded
            
            new_int_bottom_bits = new_int_bottom_bits_inc[num_bits_encoded:] + [0]*num_bits_encoded
            new_int_top_bits = new_int_top_bits_inc[num_bits_encoded:] + [1]*num_bits_encoded

            cur_interval[0] = bits2int(reversed(new_int_bottom_bits))
            cur_interval[1] = bits2int(reversed(new_int_top_bits)) + 1

            total_num_for_stats += 1
            
            # prev = torch.tensor([selection], device=device)
            prev = indices[selection_idx].view(1)
            output = torch.cat((output, prev))
            
            num_bits += num_bits_encoded
            print(prev)
            print(f"\rEncoded {num_bits}/{total_num_bits} bits")

    # out = output[len(context):].tolist()
    out = acc_output
    print(f"\nTotal bits encoded: {num_bits}")
    print(f"Encoding took {time.perf_counter() - start_time} seconds")
    return out, -total_log_probs/max(1, total_num_for_stats), total_kl/max(1, total_num_for_stats), total_num_for_stats/max(1, i), 0

# === DECODER (Tokens -> Bits) ===
# text is the list of tokens output from encoding
def decode_arithmetic(model, enc, text, context, width=90, height=90, device='cuda', temp=1.0, precision=16, topk=None):
    start_time = time.perf_counter()
    if isinstance(text, list):
        inp = torch.tensor(text, device=device, dtype=torch.long)
    else:
        inp = text
        
    if isinstance(context, list):
        context = torch.tensor(context, device=device, dtype=torch.long)

    eos_token_id = 151850
    max_val = 2**precision
    cur_interval = [0, max_val]

    prev = context
    full_seq = context
    past = None
    message = []
    
    with torch.no_grad():
        i = 0
        while i < len(inp):
            if i % (len(context) + 8193) == 0: # skip over context tokens
                print("curr:", inp[i])
                i += len(context)
                print("next:", inp[i])
                breakpoint()
                continue

            target_token = inp[i]
            
            out = model(input_ids=prev.unsqueeze(0), past_key_values=past, use_cache=True)
            logits = out.logits
            past = out.past_key_values
            
            next_token_logits = logits[0, -1, :].to(dtype=torch.float32)
            
            # Apply Mask
            mask, is_deterministic = get_gen_mask(full_seq.unsqueeze(0), next_token_logits, width, height)
            masked_logits = next_token_logits + mask

            # If deterministic, skip bit decoding
            if is_deterministic:
                prev = target_token.view(1)
                full_seq = torch.cat((full_seq, prev))

                if target_token == eos_token_id:
                    print("Finished decoding one image. Starting next...")
                    prev = context
                    full_seq = context
                    past = None
                    breakpoint()

                i += 1
                continue

            # Probabilistic Step
            logits_temp = masked_logits / temp
            probs_temp = F.softmax(logits_temp, dim=0)

            # === SYMMETRY FIX: Sort BEFORE cutoff ===
            probs_sorted, indices_sorted = torch.sort(probs_temp, descending=True)

            cur_int_range = cur_interval[1] - cur_interval[0]
            cur_threshold = 1.0 / cur_int_range
            
            # Check threshold on SORTED probs (Matches Encode)
            cutoff_indices = (probs_sorted < cur_threshold).nonzero()
            if len(cutoff_indices) > 0:
                k = max(2, cutoff_indices[0].item())
            else:
                k = len(probs_sorted)
            
            if topk:
                k = min(k, topk)
                
            # Slice sorted arrays
            probs_temp_int = probs_sorted[:k]
            indices = indices_sorted[:k]
            
            # Renormalize and Round
            probs_temp_int = probs_temp_int / probs_temp_int.sum() * cur_int_range
            probs_temp_int = probs_temp_int.round().long()
            cum_probs = probs_temp_int.cumsum(0)
            
            # Fix Overfill
            overfill_index = (cum_probs > cur_int_range).nonzero()
            if len(overfill_index) > 0:
                cum_probs = cum_probs[:overfill_index[0]]
                indices = indices[:len(cum_probs)]

            # Fix Underfill (ensure sum matches range)
            cum_probs[-1] += cur_int_range - cum_probs[-1]
            
            # Shift to absolute interval coordinates
            cum_probs += cur_interval[0]

            # === DECODE SELECTION ===
            # Find the "bin" where the target_token is located
            rank_matches = (indices == target_token).nonzero()
            if len(rank_matches) == 0:
                print(f"\nWarning: Target token {target_token} not in top-{k} at index {i}")
                print("Decoded message may be corrupted from this point.")
                break
                
            selection_idx = rank_matches[0].item()

            new_int_bottom = cum_probs[selection_idx-1] if selection_idx > 0 else cur_interval[0]
            new_int_top = cum_probs[selection_idx]

            # Emit Bits
            new_int_bottom_bits_inc = list(reversed(int2bits(new_int_bottom, precision)))
            new_int_top_bits_inc = list(reversed(int2bits(new_int_top-1, precision)))
            
            num_bits_encoded = num_same_from_beg(new_int_bottom_bits_inc, new_int_top_bits_inc)
            
            # If we are at the very end, flush the state (using bottom bits)
            if i == len(inp) - 1:
                new_bits = new_int_bottom_bits_inc
            else:
                new_bits = new_int_top_bits_inc[:num_bits_encoded]
            
            message += new_bits
            
            new_int_bottom_bits = new_int_bottom_bits_inc[num_bits_encoded:] + [0]*num_bits_encoded
            new_int_top_bits = new_int_top_bits_inc[num_bits_encoded:] + [1]*num_bits_encoded

            cur_interval[0] = bits2int(reversed(new_int_bottom_bits))
            cur_interval[1] = bits2int(reversed(new_int_top_bits)) + 1
            
            prev = target_token.view(1)
            full_seq = torch.cat((full_seq, prev))
            print(f"\rDecoded {len(message)} bits...", end="")

            i += 1

    print(f"Decoding took {time.perf_counter() - start_time} seconds")
    return message
