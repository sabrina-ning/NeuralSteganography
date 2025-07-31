import torch
import torch.nn.functional as F
from transformers import DynamicCache
import time

from utils import limit_past, kl, entropy, bits2int, int2bits, is_sent_finish, num_same_from_beg, is_cit

def encode_arithmetic(model, enc, message, context, finish_sent=False, device='cuda', temp=1.0, precision=16, topk=None):
    context = torch.tensor(context, device=device, dtype=torch.long)

    max_val = 2**precision
    cur_interval = [0, max_val] # bottom inclusive, top exclusive

    prev = context
    output = context
    past = None

    total_num_for_stats = 0
    total_log_probs = 0
    total_kl = 0 # in bits
    total_entropy_ptau = 0

    num_bits = 0

    with torch.no_grad():
        i = 0
        sent_finish = False
        while i < len(message) or (finish_sent and not sent_finish):
            print()
            # print("cache size:", past.get_seq_length() if past else 0)
            # if past:
            #     past = limit_past(past)
            out = model(input_ids=prev.unsqueeze(0), past_key_values=past, use_cache=True)
            logits = out.logits
            past = out.past_key_values

            # logits[0, -1, 151643] = -1e4 # endoftext can't happen
            # logits[0, -1, 151850] = -1e4 # endofsequence can't happen

            if not topk: # for message -> bits
                logits, indices = logits[0, -1, :].sort(descending=True)
            else: # for cover text
                logits, indices = logits[0, -1, :151643].sort(descending=True) # text-only
            
            logits = logits.double()
            logits_temp = logits / temp
            probs_temp = F.softmax(logits_temp, dim=0)
            log_probs_temp = F.log_softmax(logits_temp, dim=0)
            log_probs = F.log_softmax(logits, dim=0)
            
            # conditions for having reached the end of the message
            if i >= len(message):
                selection = 0
                sent_finish = is_sent_finish(indices[selection].item(), enc)
            else:
                # Cutoff low probabilities that would be rounded to 0
                cur_int_range = cur_interval[1]-cur_interval[0]
                cur_threshold = 1/cur_int_range
                
                cutoff_indices = (probs_temp < cur_threshold).nonzero()
                if len(cutoff_indices) > 0:
                    k = max(2, cutoff_indices[0].item())
                else:
                    k = len(probs_temp)
                    
                if topk:
                    k = min(k, topk)
                
                if not topk:
                    probs_temp_int = probs_temp[:k] # Cutoff all but top k
                else:
                    # Perform stepwise verification
                    indices = indices[:k]
                    probs = probs_temp[:k]

                    clean_indices = []
                    clean_probs = []

                    for j in range(len(indices)):
                        token_id = indices[j].item()
                        if not is_cit(enc, token_id, list(prev)):
                            clean_indices.append(token_id)
                            clean_probs.append(probs[j].item())
                    
                    if not clean_probs:
                        print("Warning: All top-k tokens were inconsistent")
                        exit

                    indices = torch.tensor(clean_indices, device=device)
                    probs_temp_int = torch.tensor(clean_probs, device=device)

                ## DEBUGGING
                # if topk:
                #     print(f"\tTop-k tokens:")
                #     for rank_idx in range(topk):
                #         token_id = indices[rank_idx].item()
                #         token_text = enc.tokenizer.decode([token_id])
                #         print(f"\t\t{rank_idx}: {[token_text, token_id]}")

                # FIXME >>>

                # Rescale to correct range
                print("interval size:", cur_int_range)
                probs_temp_int = probs_temp_int/probs_temp_int.sum()*cur_int_range
                print("probs:", probs_temp_int[:10], probs_temp_int.shape)

                # Round probabilities to integers given precision
                probs_temp_int = probs_temp_int.round().long()
                print("rounded probs:", probs_temp_int[:10], probs_temp_int.shape)
                print("k:", k)
                print("clean probs:", len(clean_probs))
                cum_probs = probs_temp_int.cumsum(0)
                print("cum probs:", cum_probs[:10], cum_probs.shape)

                # Remove any elements from the bottom if rounding caused the total prob to be too large
                overfill_index = (cum_probs > cur_int_range).nonzero()
                if len(overfill_index) > 0:
                    print("first overfill index:", overfill_index[0])
                    if overfill_index[0] == 0:
                        print("overfill!")
                        # cum_probs = torch.tensor([cur_int_range], device=device)
                    cum_probs = cum_probs[:overfill_index[0]]
                    
                # <<< FIXME numerical issue? cast to float32 temporarily
                
                # Add any mass to the top if removing/rounding causes the total prob to be too small
                cum_probs += cur_int_range-cum_probs[-1] # add

                # Get out resulting probabilities
                probs_final = cum_probs.clone()
                probs_final[1:] = cum_probs[1:] - cum_probs[:-1]

                # Convert to position in range
                cum_probs += cur_interval[0]

                # Get selected index based on binary fraction from message bits
                message_bits = message[i:i+precision]
                if i+precision > len(message):
                    message_bits = message_bits + [0]*(i+precision-len(message))
                message_idx = bits2int(reversed(message_bits))
                selection = (cum_probs > message_idx).nonzero()[0].item()
                print("message index:", message_idx)
                print("selection:", selection)

                # Calculate new range as ints
                new_int_bottom = cum_probs[selection-1] if selection > 0 else cur_interval[0]
                new_int_top = cum_probs[selection]

                # Convert range to bits
                new_int_bottom_bits_inc = list(reversed(int2bits(new_int_bottom, precision)))
                new_int_top_bits_inc = list(reversed(int2bits(new_int_top-1, precision))) # -1 here because upper bound is exclusive
                print("lower bound:", new_int_bottom, "->", new_int_bottom_bits_inc)
                print("upper bound:", new_int_top, "->", new_int_top_bits_inc)

                # Consume most significant bits which are now fixed and update interval
                num_bits_encoded = num_same_from_beg(new_int_bottom_bits_inc, new_int_top_bits_inc)
                i += num_bits_encoded

                new_int_bottom_bits = new_int_bottom_bits_inc[num_bits_encoded:] + [0]*num_bits_encoded
                new_int_top_bits = new_int_top_bits_inc[num_bits_encoded:] + [1]*num_bits_encoded

                cur_interval[0] = bits2int(reversed(new_int_bottom_bits))
                cur_interval[1] = bits2int(reversed(new_int_top_bits))+1 # +1 here because upper bound is exclusive

                cur_entropy = entropy(probs_temp, log_probs_temp)
                print('entropy:', cur_entropy)

                # Heuristic for low entropy
                # if topk and cur_entropy < 0.01:
                #     temp += 0.1
                #     print('low entropy! new temp:', temp)
                #     breakpoint()

                # Gather statistics
                total_log_probs += log_probs[selection].item()

                q = probs_final.double()/probs_final.sum()
                logq = q.log()
                total_kl += kl(q, logq, log_probs[:len(q)])
                total_entropy_ptau += entropy(probs_temp, log_probs_temp)
                total_num_for_stats += 1
            
            # Update history with new token
            prev = indices[selection].view(1)
            output = torch.cat((output, prev))

            # print("encode", enc.tokenizer.decode(prev.tolist()), f"({prev.item()})", message_bits[:num_bits_encoded])
            num_bits += num_bits_encoded
            print(num_bits)
            print()

            # For text->bits->text
            partial = enc.tokenizer.decode(output[len(context):].tolist())
            print("partial:", partial)
            if '<eos>' in partial:
                break

            # time.sleep(2)
            
    avg_NLL = -total_log_probs/total_num_for_stats
    avg_KL = total_kl/total_num_for_stats
    avg_Hq = total_entropy_ptau/total_num_for_stats
    words_per_bit = total_num_for_stats/i

    out = output[len(context):].tolist()
    print("output >>>", out)

    return out, avg_NLL, avg_KL, words_per_bit, avg_Hq

def decode_arithmetic(model, enc, text, context, device='cuda', temp=1.0, precision=16, topk=None):
    # inp is a list of token indices
    # context is a list of token indices

    if isinstance(text, str):
        inp = enc.tokenizer.encode(text)
    elif isinstance(text, list): # list -> tensor
        inp = torch.tensor(text, device=device, dtype=torch.long)
    else:
        inp = text
    print("input  >>>", inp)

    # context = torch.tensor(context, device=device, dtype=torch.long)

    max_val = 2**precision
    cur_interval = [0, max_val] # bottom inclusive, top exclusive

    num_bits = 0

    prev = torch.tensor(context, device=device, dtype=torch.long)
    past = None
    message = []
    with torch.no_grad():
        i = 0
        while i < len(inp):
            out = model(input_ids=prev.unsqueeze(0), past_key_values=past, use_cache=True)
            logits = out.logits
            past = out.past_key_values

            # logits[0, -1, 151643] = -1e4 # endoftext can't happen
            # logits[0, -1, 151850] = -1e4 # endofsequence can't happen

            if not topk: # for message -> bits
                logits, indices = logits[0, -1, :].sort(descending=True)
            else: # for cover text
                logits, indices = logits[0, -1, :151643].sort(descending=True) # text-only
            
            logits = logits.double()
            logits_temp = logits / temp
            probs_temp = F.softmax(logits_temp, dim=0)
            log_probs_temp = F.log_softmax(logits_temp, dim=0) # for entropy calculation
            
            # Cutoff low probabilities that would be rounded to 0
            cur_int_range = cur_interval[1]-cur_interval[0]
            cur_threshold = 1/cur_int_range

            cutoff_indices = (probs_temp < cur_threshold).nonzero()
            if len(cutoff_indices) > 0:
                k = max(2, cutoff_indices[0].item())
            else:
                k = len(probs_temp)
                
            if topk:
                k = min(k, topk)

            if not topk:
                probs_temp_int = probs_temp[:k] # Cutoff all but top k
            else:
                # Perform stepwise verification
                indices = indices[:k]
                probs = probs_temp[:k]

                clean_indices = []
                clean_probs = []

                for j in range(len(indices)):
                    token_id = indices[j].item()
                    if not is_cit(enc, token_id, list(prev)):
                        clean_indices.append(token_id)
                        clean_probs.append(probs[j].item())
                
                if not clean_probs:
                    print("Warning: All top-k tokens were inconsistent")
                    exit

                indices = torch.tensor(clean_indices, device=device) # FIXME ??
                probs_temp_int = torch.tensor(clean_probs, device=device)
        
            ## DEBUGGING
            # if topk:
            # print(f"\tTop-k tokens:")
            # for rank_idx in range(10):
            #     token_id = indices[rank_idx].item()
            #     token_text = enc.tokenizer.decode([token_id])
            #     print(f"\t\t{rank_idx}: {[token_text, token_id]}")

            # Rescale to correct range
            probs_temp_int = probs_temp_int/probs_temp_int.sum()*cur_int_range

            # Round probabilities to integers given precision
            probs_temp_int = probs_temp_int.round().long()
            cum_probs = probs_temp_int.cumsum(0)

            # Remove any elements from the bottom if rounding caused the total prob to be too large
            overfill_index = (cum_probs > cur_int_range).nonzero()
            if len(overfill_index) > 0:
                # if topk and overfill_index[0] == 0:
                #     print("overfill -> entropy:", entropy(probs_temp, log_probs_temp))
                #     temp = 1.3
                #     continue
                cum_probs = cum_probs[:overfill_index[0]]
                k = overfill_index[0].item()

            # Add any mass to the top if removing/rounding causes the total prob to be too small
            cum_probs += cur_int_range-cum_probs[-1] # add

            # Convert to position in range
            cum_probs += cur_interval[0]

            rank = (indices == inp[i]).nonzero().item()

            if rank >= k:
                print('Error: tokenization inconsistency, rank >= k')
            
            selection = rank
            
            # Calculate new range as ints
            new_int_bottom = cum_probs[selection-1] if selection > 0 else cur_interval[0]
            new_int_top = cum_probs[selection]

            # Convert range to bits
            new_int_bottom_bits_inc = list(reversed(int2bits(new_int_bottom, precision)))
            new_int_top_bits_inc = list(reversed(int2bits(new_int_top-1, precision))) # -1 here because upper bound is exclusive
            
            # Emit most significant bits which are now fixed and update interval
            num_bits_encoded = num_same_from_beg(new_int_bottom_bits_inc, new_int_top_bits_inc)
            if i == len(inp)-1:
                new_bits = new_int_bottom_bits_inc
            else:
                new_bits = new_int_top_bits_inc[:num_bits_encoded]
            message += new_bits

            new_int_bottom_bits = new_int_bottom_bits_inc[num_bits_encoded:] + [0]*num_bits_encoded
            new_int_top_bits = new_int_top_bits_inc[num_bits_encoded:] + [1]*num_bits_encoded

            cur_interval[0] = bits2int(reversed(new_int_bottom_bits))
            cur_interval[1] = bits2int(reversed(new_int_top_bits))+1 # +1 here because upper bound is exclusive

            cur_entropy = entropy(probs_temp, log_probs_temp)
            # print(cur_entropy)

            # Heuristic for low entropy
            # if topk and cur_entropy < 0.01:
            #     print('low entropy!', cur_entropy)
            #     temp = min(temp + 0.1, 1.3)
            # elif topk:
            #     temp = 0.9
            # print()
            
            # Update history with new token
            # prev = torch.tensor([inp[i]], device=device, dtype=torch.long)
            prev = torch.tensor([indices[selection].item()], device=device, dtype=torch.long)

            # print("decode", enc.tokenizer.decode([inp[i]]), f"({inp[i]})", new_bits)
            num_bits += num_bits_encoded
            print(num_bits)
            # print()
            
            i += 1

    return message