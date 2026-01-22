import torch
import torch.nn.functional as F
from transformers import DynamicCache
import random
# import time
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
        mask[pad_token_id] = 0.0
    else: # continue visual tokens
        mask[visual_start_id:] = 0.0
        is_deterministic = False
        
    return mask, is_deterministic

    # === ENCODER (Bits -> Tokens) ===

# message is a list of bits
def encode_arithmetic(model, enc, message, context, width=90, height=90, finish_sent=False, device='cuda', temp=1.0, precision=16, topk=None):
    # print("message:", message)
    
    if isinstance(context, list):
        context = torch.tensor(context, device=device, dtype=torch.long)

    # Correct EOS ID for stopping condition
    eos_token_id = 151850

    max_val = 2**precision
    cur_interval = [0, max_val] 

    prev = context
    output = context
    past = None
    
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
                    break
                continue

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
            print(f"\rEncoded {num_bits} bits...", end="")

    out = output[len(context):].tolist()
    print(f"\nTotal bits encoded: {num_bits}")
    return out, -total_log_probs/max(1, total_num_for_stats), total_kl/max(1, total_num_for_stats), total_num_for_stats/max(1, i), 0

# === DECODER (Tokens -> Bits) ===
def decode_arithmetic(model, enc, text, context, width=90, height=90, device='cuda', temp=1.0, precision=16, topk=None):
    if isinstance(text, list):
        inp = torch.tensor(text, device=device, dtype=torch.long)
    else:
        inp = text
        
    if isinstance(context, list):
        context = torch.tensor(context, device=device, dtype=torch.long)

    max_val = 2**precision
    cur_interval = [0, max_val]

    prev = context
    full_seq = context
    past = None
    message = []
    
    with torch.no_grad():
        for i in range(len(inp)):
            target_token = inp[i]
            
            out = model(input_ids=prev.unsqueeze(0), past_key_values=past, use_cache=True)
            logits = out.logits
            past = out.past_key_values
            
            next_token_logits = logits[0, -1, :].to(dtype=torch.float32)
            
            # 1. Apply Mask
            mask, is_deterministic = get_gen_mask(full_seq.unsqueeze(0), next_token_logits, width, height)
            masked_logits = next_token_logits + mask

            # If deterministic, skip bit decoding
            if is_deterministic:
                prev = target_token.view(1)
                full_seq = torch.cat((full_seq, prev))
                continue

            # 2. Probabilistic Step
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

    print()
    return message

# def encode_arithmetic(model, enc, message, context, finish_sent=False, device='cuda', temp=1.0, precision=16, topk=None):
#     context = torch.tensor(context, device=device, dtype=torch.long)

#     max_val = 2**precision
#     cur_interval = [0, max_val] # bottom inclusive, top exclusive

#     prev = context
#     output = context
#     past = None

#     total_num_for_stats = 0
#     total_log_probs = 0
#     total_kl = 0 # in bits
#     total_entropy_ptau = 0

#     num_bits = 0
    
#     # probs_over_time = []
#     # entropy_over_time = []
#     # token_labels = []

#     with torch.no_grad():
#         i = 0
#         sent_finish = False
#         while i < len(message) or (finish_sent and not sent_finish):
#             out = model(input_ids=prev.unsqueeze(0), past_key_values=past, use_cache=True)
#             logits = out.logits
#             past = out.past_key_values

#             # logits[0, -1, 151643] = -1e4 # endoftext can't happen
#             # logits[0, -1, 151850] = -1e4 # endofsequence can't happen

#             if not topk: # for message -> bits
#                 logits, indices = logits[0, -1, :].sort(descending=True)
#             else: # for cover text
#                 logits, indices = logits[0, -1, :151643].sort(descending=True) # text-only
            
#             logits = logits.double()
#             logits_temp = logits / temp
#             probs_temp = F.softmax(logits_temp, dim=0)
#             log_probs_temp = F.log_softmax(logits_temp, dim=0)
#             log_probs = F.log_softmax(logits, dim=0)
            
#             # conditions for having reached the end of the message
#             if i >= len(message):
#                 selection = 0
#                 sent_finish = is_sent_finish(indices[selection].item(), enc)
#             else:
#                 # Cutoff low probabilities that would be rounded to 0
#                 cur_int_range = cur_interval[1]-cur_interval[0]
#                 cur_threshold = 1/cur_int_range
                
#                 cutoff_indices = (probs_temp < cur_threshold).nonzero()
#                 if len(cutoff_indices) > 0:
#                     k = max(2, cutoff_indices[0].item())
#                 else:
#                     k = len(probs_temp)
                    
#                 if topk:
#                     k = min(k, topk)
                
#                 if not topk:
#                     probs_temp_int = probs_temp[:k] # Cutoff all but top k
#                 else:
#                     # Perform stepwise verification
#                     indices = indices[:k]
#                     probs = probs_temp[:k]

#                     clean_indices = []
#                     clean_probs = []

#                     for j in range(len(indices)):
#                         token_id = indices[j].item()
#                         if not is_cit(enc, token_id, list(prev)):
#                             clean_indices.append(token_id)
#                             clean_probs.append(probs[j].item())
                    
#                     if not clean_probs:
#                         print("Warning: All top-k tokens were inconsistent")
#                         exit

#                     indices = torch.tensor(clean_indices, device=device)
#                     probs_temp_int = torch.tensor(clean_probs, device=device)

#                 ## DEBUGGING
#                 # if topk:
#                 #     print(f"\tTop-k tokens:")
#                 #     for rank_idx in range(topk):
#                 #         token_id = indices[rank_idx].item()
#                 #         token_text = enc.tokenizer.decode([token_id])
#                 #         print(f"\t\t{rank_idx}: {[token_text, token_id]}")

#                 # FIXME >>>

#                 # Rescale to correct range
#                 print("interval size:", cur_int_range)
#                 print("probs:", probs_temp_int[:10], probs_temp_int.shape)

#                 # top_probs = probs_temp_int[:10].tolist()
#                 # top_probs += [0] * (10 - len(top_probs))
#                 # probs_over_time.append(top_probs)
#                 # if not token_labels:
#                 #     token_labels = [i + 1 for i in range(10)]

#                 probs_temp_int = probs_temp_int/probs_temp_int.sum()*cur_int_range

#                 # Round probabilities to integers given precision
#                 probs_temp_int = probs_temp_int.round().long()
#                 # print("rounded probs:", probs_temp_int[:10], probs_temp_int.shape)
#                 # print("k:", k)
#                 # print("clean probs:", len(clean_probs))
#                 cum_probs = probs_temp_int.cumsum(0)
#                 # print("cum probs:", cum_probs[:10], cum_probs.shape)

#                 # Remove any elements from the bottom if rounding caused the total prob to be too large
#                 overfill_index = (cum_probs > cur_int_range).nonzero()
#                 if len(overfill_index) > 0:
#                     print("first overfill index:", overfill_index[0])
#                     if overfill_index[0] == 0:
#                         print("overfill!")
#                         # cum_probs = torch.tensor([cur_int_range], device=device)
#                     cum_probs = cum_probs[:overfill_index[0]]
                    
#                 # <<< FIXME numerical issue? cast to float32 temporarily
                
#                 # Add any mass to the top if removing/rounding causes the total prob to be too small
#                 # print(type(cum_probs))
#                 cum_probs[-1] += cur_int_range-cum_probs[-1] # add

#                 # Get out resulting probabilities
#                 probs_final = cum_probs.clone()
#                 probs_final[1:] = cum_probs[1:] - cum_probs[:-1]

#                 # Convert to position in range
#                 cum_probs += cur_interval[0]

#                 # Get selected index based on binary fraction from message bits
#                 message_bits = message[i:i+precision]
#                 if i+precision > len(message):
#                     message_bits = message_bits + [0]*(i+precision-len(message))
#                 message_idx = bits2int(reversed(message_bits))
#                 selection = (cum_probs > message_idx).nonzero()[0].item()
#                 # print("message index:", message_idx)
#                 # print("selection:", selection)

#                 # Calculate new range as ints
#                 new_int_bottom = cum_probs[selection-1] if selection > 0 else cur_interval[0]
#                 new_int_top = cum_probs[selection]

#                 # Convert range to bits
#                 new_int_bottom_bits_inc = list(reversed(int2bits(new_int_bottom, precision)))
#                 new_int_top_bits_inc = list(reversed(int2bits(new_int_top-1, precision))) # -1 here because upper bound is exclusive
#                 # print("lower bound:", new_int_bottom, "->", new_int_bottom_bits_inc)
#                 # print("upper bound:", new_int_top, "->", new_int_top_bits_inc)

#                 # Consume most significant bits which are now fixed and update interval
#                 num_bits_encoded = num_same_from_beg(new_int_bottom_bits_inc, new_int_top_bits_inc)
#                 i += num_bits_encoded

#                 new_int_bottom_bits = new_int_bottom_bits_inc[num_bits_encoded:] + [0]*num_bits_encoded
#                 new_int_top_bits = new_int_top_bits_inc[num_bits_encoded:] + [1]*num_bits_encoded

#                 cur_interval[0] = bits2int(reversed(new_int_bottom_bits))
#                 cur_interval[1] = bits2int(reversed(new_int_top_bits))+1 # +1 here because upper bound is exclusive

#                 cur_entropy = entropy(probs_temp, log_probs_temp)
#                 # print('entropy:', cur_entropy)

#                 # entropy_over_time.append(cur_entropy)

#                 # Heuristic for low entropy
#                 # if topk and cur_entropy < 0.01:
#                 #     temp += 0.1
#                 #     print('low entropy! new temp:', temp)
#                     # breakpoint()

#                 # Gather statistics
#                 total_log_probs += log_probs[selection].item()

#                 q = probs_final.double()/probs_final.sum()
#                 logq = q.log()
#                 total_kl += kl(q, logq, log_probs[:len(q)])
#                 total_entropy_ptau += entropy(probs_temp, log_probs_temp)
#                 total_num_for_stats += 1
            
#             # Update history with new token
#             prev = indices[selection].view(1)
#             output = torch.cat((output, prev))

#             # print("encode", enc.tokenizer.decode(prev.tolist()), f"({prev.item()})", message_bits[:num_bits_encoded])
#             num_bits += num_bits_encoded
#             print(num_bits)
#             print()

#             # Heuristic for long contexts
#             # print("output len:", len(output))
#             # if len(output[len(context):]) % 200 == 0:
#             #     prev = output[-200:]
#             #     past = None
#             #     breakpoint()

#             # For text->bits->text
#             partial = enc.tokenizer.decode(output[len(context):].tolist())
#             print("partial:", partial)
#             if '<eos>' in partial:
#                 break

#             # time.sleep(2)

#     # # Plot entropy over time
#     # plt.figure(figsize=(10, 4))
#     # plt.plot(entropy_over_time, label='Entropy')
#     # plt.xlabel('Step')
#     # plt.ylabel('Entropy')
#     # plt.title('Entropy over time')
#     # plt.legend()
#     # plt.grid(True)
#     # plt.tight_layout()
#     # plt.savefig("plot_entropy.png")
#     # plt.close

#     # # Plot probs_temp_int for top-10 tokens
#     # plt.figure(figsize=(12, 6))
#     # probs_array = list(zip(*probs_over_time))
#     # for i, probs in enumerate(probs_array):
#     #     plt.plot(probs, label=f'Token {i}: {token_labels[i]}')
#     # plt.xlabel('Step')
#     # plt.ylabel('Rounded Probability')
#     # plt.title('Top-10 token probabilities over time')
#     # plt.legend()
#     # plt.grid(True)
#     # plt.tight_layout()
#     # plt.savefig("plot_probs.png")
#     # plt.close

#     avg_NLL = -total_log_probs/total_num_for_stats
#     avg_KL = total_kl/total_num_for_stats
#     avg_Hq = total_entropy_ptau/total_num_for_stats
#     words_per_bit = total_num_for_stats/i

#     out = output[len(context):].tolist()
#     print("output >>>", out)

#     return out, avg_NLL, avg_KL, words_per_bit, avg_Hq

# def decode_arithmetic(model, enc, text, context, device='cuda', temp=1.0, precision=16, topk=None):
#     # inp is a list of token indices
#     # context is a list of token indices

#     if isinstance(text, str):
#         inp = enc.tokenizer.encode(text)
#     elif isinstance(text, list): # list -> tensor
#         inp = torch.tensor(text, device=device, dtype=torch.long)
#     else:
#         inp = text
#     print("input  >>>", inp)

#     # context = torch.tensor(context, device=device, dtype=torch.long)

#     max_val = 2**precision
#     cur_interval = [0, max_val] # bottom inclusive, top exclusive

#     num_bits = 0

#     prev = torch.tensor(context, device=device, dtype=torch.long)
#     past = None
#     message = []
#     with torch.no_grad():
#         i = 0
#         while i < len(inp):
#             out = model(input_ids=prev.unsqueeze(0), past_key_values=past, use_cache=True)
#             logits = out.logits
#             past = out.past_key_values

#             # logits[0, -1, 151643] = -1e4 # endoftext can't happen
#             # logits[0, -1, 151850] = -1e4 # endofsequence can't happen

#             if not topk: # for message -> bits
#                 logits, indices = logits[0, -1, :].sort(descending=True)
#             else: # for cover text
#                 logits, indices = logits[0, -1, :151643].sort(descending=True) # text-only
            
#             logits = logits.double()
#             logits_temp = logits / temp
#             probs_temp = F.softmax(logits_temp, dim=0)
#             log_probs_temp = F.log_softmax(logits_temp, dim=0) # for entropy calculation
            
#             # Cutoff low probabilities that would be rounded to 0
#             cur_int_range = cur_interval[1]-cur_interval[0]
#             cur_threshold = 1/cur_int_range

#             cutoff_indices = (probs_temp < cur_threshold).nonzero()
#             if len(cutoff_indices) > 0:
#                 k = max(2, cutoff_indices[0].item())
#             else:
#                 k = len(probs_temp)
                
#             if topk:
#                 k = min(k, topk)

#             if not topk:
#                 probs_temp_int = probs_temp[:k] # Cutoff all but top k
#             else:
#                 # Perform stepwise verification
#                 indices = indices[:k]
#                 probs = probs_temp[:k]

#                 clean_indices = []
#                 clean_probs = []

#                 for j in range(len(indices)):
#                     token_id = indices[j].item()
#                     if not is_cit(enc, token_id, list(prev)):
#                         clean_indices.append(token_id)
#                         clean_probs.append(probs[j].item())
                
#                 if not clean_probs:
#                     print("Warning: All top-k tokens were inconsistent")
#                     exit

#                 indices = torch.tensor(clean_indices, device=device) # FIXME ??
#                 probs_temp_int = torch.tensor(clean_probs, device=device)
        
#             ## DEBUGGING
#             # if topk:
#             # print(f"\tTop-k tokens:")
#             # for rank_idx in range(10):
#             #     token_id = indices[rank_idx].item()
#             #     token_text = enc.tokenizer.decode([token_id])
#             #     print(f"\t\t{rank_idx}: {[token_text, token_id]}")

#             # Rescale to correct range
#             probs_temp_int = probs_temp_int/probs_temp_int.sum()*cur_int_range

#             # Round probabilities to integers given precision
#             probs_temp_int = probs_temp_int.round().long()
#             cum_probs = probs_temp_int.cumsum(0)

#             # Remove any elements from the bottom if rounding caused the total prob to be too large
#             overfill_index = (cum_probs > cur_int_range).nonzero()
#             if len(overfill_index) > 0:
#                 # if topk and overfill_index[0] == 0:
#                 #     print("overfill -> entropy:", entropy(probs_temp, log_probs_temp))
#                 #     temp = 1.3
#                 #     continue
#                 cum_probs = cum_probs[:overfill_index[0]]
#                 k = overfill_index[0].item()

#             # Add any mass to the top if removing/rounding causes the total prob to be too small
#             cum_probs[-1] += cur_int_range-cum_probs[-1] # add

#             # Convert to position in range
#             cum_probs += cur_interval[0]

#             rank = (indices == inp[i]).nonzero().item()

#             if rank >= k:
#                 print(rank)
#                 print(k)
#                 print('Error: tokenization inconsistency, rank >= k')
            
#             selection = rank
            
#             # Calculate new range as ints
#             new_int_bottom = cum_probs[selection-1] if selection > 0 else cur_interval[0]
#             new_int_top = cum_probs[selection]

#             # Convert range to bits
#             new_int_bottom_bits_inc = list(reversed(int2bits(new_int_bottom, precision)))
#             new_int_top_bits_inc = list(reversed(int2bits(new_int_top-1, precision))) # -1 here because upper bound is exclusive
            
#             # Emit most significant bits which are now fixed and update interval
#             num_bits_encoded = num_same_from_beg(new_int_bottom_bits_inc, new_int_top_bits_inc)
#             if i == len(inp)-1:
#                 new_bits = new_int_bottom_bits_inc
#             else:
#                 new_bits = new_int_top_bits_inc[:num_bits_encoded]
#             message += new_bits

#             new_int_bottom_bits = new_int_bottom_bits_inc[num_bits_encoded:] + [0]*num_bits_encoded
#             new_int_top_bits = new_int_top_bits_inc[num_bits_encoded:] + [1]*num_bits_encoded

#             cur_interval[0] = bits2int(reversed(new_int_bottom_bits))
#             cur_interval[1] = bits2int(reversed(new_int_top_bits))+1 # +1 here because upper bound is exclusive

#             cur_entropy = entropy(probs_temp, log_probs_temp)
#             # print(cur_entropy)

#             # Heuristic for low entropy
#             # if topk and cur_entropy < 0.01:
#             #     temp += 0.1
#             #     print('low entropy! new temp:', temp)
#             # elif topk:
#             #     temp = 0.9
#             # print()
            
#             # Update history with new token
#             # prev = torch.tensor([inp[i]], device=device, dtype=torch.long)
#             prev = torch.tensor([indices[selection].item()], device=device, dtype=torch.long)

#             # print("decode", enc.tokenizer.decode([inp[i]]), f"({inp[i]})", new_bits)
#             num_bits += num_bits_encoded
#             print(num_bits)
#             # print()
            
#             i += 1

#     return message