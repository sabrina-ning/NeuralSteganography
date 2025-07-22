import torch
import torch.nn.functional as F
from transformers import DynamicCache

from utils import limit_past, kl, entropy, bits2int, int2bits, is_sent_finish, num_same_from_beg, is_cit

# topk=None means all tokens in vocab should be considered (for message -> uniform bits)
def encode_arithmetic(model, enc, message, context, finish_sent=False, device='cuda', temp=1.0, precision=16, topk=None):
    context = torch.tensor(context, device=device, dtype=torch.long)

    max_val = 2**precision
    # threshold = 2**(-precision)
    cur_interval = [0, max_val] # bottom inclusive, top exclusive

    prev = context
    output = context
    past = None

    # total_num = 0
    total_num_for_stats = 0
    total_log_probs = 0
    total_kl = 0 # in bits
    total_entropy_ptau = 0
    # total_num_sents = 0

    ranks = []

    with torch.no_grad():
        i = 0
        sent_finish = False
        while i < len(message) or (finish_sent and not sent_finish):
            out = model(input_ids=prev.unsqueeze(0), past_key_values=DynamicCache.from_legacy_cache(past), use_cache=True)
            logits = out.logits
            past = out.past_key_values

            if gpt2:
                past = limit_past(past)
                logits[0, -1, 50256] = -1e4
                logits[0, -1, 628] = -1e4
            else: # emu3
                logits[0, -1, 151643] = -1e4 # endoftext can't happen
                # logits[0, -1, 271] = -1e4 # 2 newlines can't happen
                logits[0, -1, 151850] = -1e4 # endofsequence token can't happen

            if gpt2:
                logits, indices = logits[0, -1, :].sort(descending=True)
            else:
                logits, indices = logits[0, -1, :151643].sort(descending=True)
            
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
                
                # probs_temp_int = probs_temp[:k] # Cutoff all but top k

                if not topk:
                    probs_temp_int = probs_temp[:k] # Cutoff all but top k
                else:
                    # Perform stepwise verification
                    indices = indices[:k]
                    probs = probs_temp[:k]

                    clean_indices = []
                    clean_probs = []
                    prev_list = list(prev)

                    for j in range(len(indices)):
                        token_id = indices[j].item()
                        # print(len(prev_list))
                        if not is_cit(enc, token_id, prev_list.copy()):
                            clean_indices.append(token_id)
                            clean_probs.append(probs[j].item())
                    
                    if not clean_probs:
                        print("Warning: All top-k tokens were inconsistent")
                        exit

                    probs_temp_int = torch.tensor(clean_probs, device=device)

                # DEBUGGING >>>
                # print(f"\tTop-k tokens:")
                # for rank_idx in range(300):
                # token_id = indices[rank_idx].item()
                # token_text = enc.decode([token_id])
                # print(f"\t\t{rank_idx}: {token_text} ({token_id})")

                # Rescale to correct range
                probs_temp_int = probs_temp_int/probs_temp_int.sum()*cur_int_range

                # Round probabilities to integers given precision
                probs_temp_int = probs_temp_int.round().long()
                cum_probs = probs_temp_int.cumsum(0)

                # Remove any elements from the bottom if rounding caused the total prob to be too large
                overfill_index = (cum_probs > cur_int_range).nonzero()
                if len(overfill_index) > 0:
                    cum_probs = cum_probs[:overfill_index[0]]

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

                # Calculate new range as ints
                new_int_bottom = cum_probs[selection-1] if selection > 0 else cur_interval[0]
                new_int_top = cum_probs[selection]

                # Convert range to bits
                new_int_bottom_bits_inc = list(reversed(int2bits(new_int_bottom, precision)))
                new_int_top_bits_inc = list(reversed(int2bits(new_int_top-1, precision))) # -1 here because upper bound is exclusive

                # Consume most significant bits which are now fixed and update interval
                num_bits_encoded = num_same_from_beg(new_int_bottom_bits_inc, new_int_top_bits_inc)
                i += num_bits_encoded

                new_int_bottom_bits = new_int_bottom_bits_inc[num_bits_encoded:] + [0]*num_bits_encoded
                new_int_top_bits = new_int_top_bits_inc[num_bits_encoded:] + [1]*num_bits_encoded

                cur_interval[0] = bits2int(reversed(new_int_bottom_bits))
                cur_interval[1] = bits2int(reversed(new_int_top_bits))+1 # +1 here because upper bound is exclusive

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
            # total_num += 1

            ranks.append(selection)

            print("encode", enc.decode(prev.tolist()), f"({prev.item()})", message_bits[:num_bits_encoded])
            # print(enc.decode(prev.tolist()), f"({prev.item()}): selection = {selection}\n")

            # For text->bits->text
            partial = enc.decode(output[len(context):].tolist())
            if '<eos>' in partial:
                print('BREAK')
                break
            
    avg_NLL = -total_log_probs/total_num_for_stats
    avg_KL = total_kl/total_num_for_stats
    avg_Hq = total_entropy_ptau/total_num_for_stats
    words_per_bit = total_num_for_stats/i

    print("encoded >>>", output[len(context):].tolist())
    print(ranks)
    return output[len(context):].tolist(), avg_NLL, avg_KL, words_per_bit, avg_Hq

def decode_arithmetic(model, enc, text, context, device='cuda', temp=1.0, precision=16, topk=None, gpt2=False):
    # inp is a list of token indices
    # context is a list of token indices

    # inp = enc.encode(text)
    if isinstance(text, str):
        inp = enc.encode(text)
    else:
        # print("decoded >>>", text)
        inp = torch.tensor(text, device=device, dtype=torch.long)
    
    # i = 0
    # two_newlines_id = 271
    # if gpt2:
    #     two_newlines_id = 628
    # while i < len(inp):
    #     if inp[i] == two_newlines_id:
    #         inp[i] = 198
    #         inp[i+1:i+1] = [198]
    #         i += 2
    #         print(f"Fixed common BPE error: 198, 198 (2 newlines) interpretted as {two_newlines_id} (2 newlines)")
    #     else:
    #         i += 1

    print("decoded >>>", inp)    

    context = torch.tensor(context, device=device, dtype=torch.long)
    # print(context)

    max_val = 2**precision
    # threshold = 2**(-precision)
    cur_interval = [0, max_val] # bottom inclusive, top exclusive

    ranks = []

    prev = context
    past = None
    message = []
    with torch.no_grad():
        i = 0
        while i < len(inp):
            out = model(input_ids=prev.unsqueeze(0), past_key_values=DynamicCache.from_legacy_cache(past), use_cache=True)
            logits = out.logits
            past = out.past_key_values

            if gpt2:
                past = limit_past(past)
                logits[0, -1, 50256] = -1e4
                logits[0, -1, 628] = -1e4
            else:
                logits[0, -1, 151643] = -1e4 # endoftext can't happen
                # logits[0, -1, 271] = -1e4 # 2 newlines can't happen
                logits[0, -1, 151850] = -1e4 # endofsequence token can't happen

            if gpt2:
                logits, indices = logits[0, -1, :].sort(descending=True)
            else:
                logits, indices = logits[0, -1, :151643].sort(descending=True)
            
            logits = logits.double()
            logits_temp = logits / temp
            probs_temp = F.softmax(logits_temp, dim=0)
            
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
            
            # probs_temp_int = probs_temp[:k] # Cutoff all but top k

            if not topk:
                probs_temp_int = probs_temp[:k] # Cutoff all but top k
            else:
                # Perform stepwise verification
                indices = indices[:k]
                probs = probs_temp[:k]

                clean_indices = []
                clean_probs = []
                prev_list = list(prev)

                for j in range(len(indices)):
                    token_id = indices[j].item()
                    if not is_cit(enc, token_id, prev_list.copy()):
                        clean_indices.append(token_id)
                        clean_probs.append(probs[j].item())
                
                if not clean_probs:
                    print("Warning: All top-k tokens were inconsistent")
                    exit

                probs_temp_int = torch.tensor(clean_probs, device=device)
        
            # DEBUGGING >>>
            # print(f"\tTop-k tokens:")
            # for rank_idx in range(300):
            # token_id = indices[rank_idx].item()
            # token_text = enc.decode([token_id])
            # print(f"\t\t{rank_idx}: {token_text} ({token_id})")

            # Rescale to correct range
            probs_temp_int = probs_temp_int/probs_temp_int.sum()*cur_int_range

            # Round probabilities to integers given precision
            probs_temp_int = probs_temp_int.round().long()
            cum_probs = probs_temp_int.cumsum(0)

            # Remove any elements from the bottom if rounding caused the total prob to be too large
            overfill_index = (cum_probs > cur_int_range).nonzero()
            if len(overfill_index) > 0:
                cum_probs = cum_probs[:overfill_index[0]]
                k = overfill_index[0].item()

            # Add any mass to the top if removing/rounding causes the total prob to be too small
            cum_probs += cur_int_range-cum_probs[-1] # add

            # Convert to position in range
            cum_probs += cur_interval[0]

            rank = (indices == inp[i]).nonzero().item()

            # # Handle most errors that could happen because of BPE with heuristic
            if rank >= k:
                print(ranks, rank)
                print('ERROR')
            #     true_token_text = enc.decode([inp[i]])
            #     for rank_idx in range(k):
            #         prop_token_id = indices[rank_idx].item()
            #         prop_token_text = enc.decode([prop_token_id])
            #         # common case that is not caught
            #         # if inp[i] == 128 and indices[rank_idx] == 198:
            #         #     rank = rank_idx
            #         #     inp[i] = indices[rank_idx].item()
            #         #     break
                    
            #         # Is there a more likely prefix token that could be the actual token generated?
            #         if len(prop_token_text) <= len(true_token_text) and \
            #                 prop_token_text == true_token_text[:len(prop_token_text)]:
            #             rank = rank_idx
            #             suffix = true_token_text[len(prop_token_text):]
            #             suffix_tokens = enc.encode(suffix) # a list
            #             # suffix_tokens = encode(enc, suffix)
            #             inp[i] = prop_token_id
            #             inp[i+1:i+1] = suffix_tokens # insert suffix tokens into list
            #             print('shorter')
            #             break

            #         # Is there a more likely longer token that could be the actual token generated?
            #         elif len(prop_token_text) > len(true_token_text) and \
            #                   true_token_text == prop_token_text[:len(true_token_text)]:
            #             whole_text = true_token_text
            #             num_extra = 1
            #             while len(whole_text) < len(prop_token_text) and (i+num_extra) < len(inp):
            #                 whole_text += enc.decode([inp[i+num_extra]])
            #                 num_extra += 1
            #             if prop_token_text == whole_text[:len(prop_token_text)]:
            #                 rank = rank_idx
            #                 inp[i] = prop_token_id
            #                 for j in range(1, num_extra):
            #                     del inp[i+j]

            #                 if len(whole_text) > len(prop_token_text):
            #                     suffix = whole_text[len(prop_token_text):]
            #                     suffix_tokens = enc.encode(suffix) # a list
            #                     # suffix_tokens = encode(enc, suffix)
            #                     inp[i+1:i+1] = suffix_tokens # insert suffix tokens into list
            #                 print('longer')
            #                 break
            #     else:
            #         # print('Unable to fix BPE error: token received: %s=%d, text: %s' % (true_token_text, inp[i], text))
            #         print('Unable to fix BPE error: token received: %s=%d' % (true_token_text, inp[i]))
            #         rank = 0

            # if not topk: # only for message -> uniform bits
            #     rank = (indices == inp[i]).nonzero().item()
            #     num_matched = 1
            # else:
            #     # Heuristic search for correct tokenization
            #     candidates = []
            #     for rank_idx in range(k):
            #         predicted_token_id = indices[rank_idx].item()
            #         predicted_token_text = enc.decode([predicted_token_id])

            #         retokenized_ids = enc.encode(predicted_token_text)
            #         # j = 0
            #         # while j < len(retokenized_ids):
            #         #     if retokenized_ids[j] == two_newlines_id:
            #         #         retokenized_ids[j] = 198
            #         #         retokenized_ids[j+1:j+1] = [198]
            #         #         j += 2
            #         #         # print(f"FIXED {two_newlines_id} -> 198, 198")
            #         #     else:
            #         #         j += 1

            #         # Check if retokenization matches the input
            #         match = True
            #         if len(retokenized_ids) > len(inp) - i:
            #             match = False
            #         else:
            #             for j in range(len(retokenized_ids)):
            #                 if retokenized_ids[j] != inp[i+j]:
            #                     match = False
            #                     break
            #             # Common case: inconsistent parsing of multiple newlines
            #             if not match:
            #                 predicted_token_text_new = predicted_token_text + "A"
            #                 retokenized_ids_new = enc.encode(predicted_token_text_new)
            #                 if retokenized_ids_new[-1] == enc.encode("A")[0]:
            #                     match = True
            #                     for j in range(len(retokenized_ids_new)-1): # last token id corresponds to "A"
            #                         if retokenized_ids_new[j] != inp[i+j]:
            #                             match = False
            #                             break
            #                     if match:
            #                         print("retokenized:", retokenized_ids, retokenized_ids_new[:-1])
            #                         retokenized_ids = retokenized_ids_new[:-1]

            #         if match:
            #             candidates.append({'rank': rank_idx, 'num_matched': len(retokenized_ids)})

            #     if not candidates:
            #         print(f"Heuristic search failed: No candidate found for input starting with [{enc.decode([inp[i]])}]")
            #         rank_idx = (indices == inp[i]).nonzero().item()
            #         rank = rank_idx if rank_idx < k else 0
            #         num_matched = 1
            #     else:
            #         print(candidates)
            #         # Greedy selection: choose candidate that matches the most tokens; if tie, then choose one with lower rank
            #         best_candidate = max(candidates, key=lambda candidate: (candidate['num_matched'], -candidate['rank']))
            #         rank = best_candidate['rank']
            #         num_matched = best_candidate['num_matched']
            #         if len(candidates) > 1:
            #             print("chose: rank", rank)
            
            selection = rank
            # print(selection)
            ranks.append(selection)
            
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
            
            # Update history with new token
            # prev = torch.tensor([inp[i]], device=device, dtype=torch.long)
            prev = torch.tensor([indices[selection].item()], device=device, dtype=torch.long)

            print("decode", enc.decode([inp[i]]), f"({inp[i]})", new_bits)
            # print(enc.decode([inp[i]]), f"({inp[i]}): selection = {selection}\n")

            # i += num_matched
            i += 1

    print(ranks)
    # print("new decoded >>>", inp)  
    
    return message