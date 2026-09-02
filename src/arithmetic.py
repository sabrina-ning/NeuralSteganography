"""Text-mode arithmetic coding for neural steganography.

Encodes a secret bit-string into natural-looking cover text using an autoregressive
language model as the entropy source. Decoding recovers the original bits from the
cover text using the same model and context.

This is the core text-only steganography algorithm from the original STEGASURAS paper,
updated for compatibility with modern HuggingFace models (GPT-2, Qwen, Emu3).
"""

import torch
import torch.nn.functional as F
from transformers import DynamicCache
import time

from src.utils import limit_past, kl, entropy, bits2int, int2bits, is_sent_finish, num_same_from_beg, is_cit


def encode_arithmetic(model, enc, message, context, finish_sent=False, device='cuda', temp=1.0, precision=16, topk=None):
    """Encode a bit-string into cover text tokens using arithmetic coding.

    Args:
        model: autoregressive language model
        enc: processor/tokenizer
        message: list of bits (0/1) to encode
        context: list of context token ids
        finish_sent: if True, continue generating until sentence-ending punctuation
        temp: sampling temperature
        precision: arithmetic coder precision in bits
        topk: if set, restrict generation to text-only tokens and apply stepwise verification
    Returns:
        (tokens, avg_NLL, avg_KL, words_per_bit, avg_entropy)
    """
    context = torch.tensor(context, device=device, dtype=torch.long)

    max_val = 2 ** precision
    cur_interval = [0, max_val]  # bottom inclusive, top exclusive

    prev = context
    output = context
    past = None

    total_num_for_stats = 0
    total_log_probs = 0
    total_kl = 0
    total_entropy_ptau = 0
    num_bits = 0

    with torch.no_grad():
        i = 0
        sent_finish = False
        while i < len(message) or (finish_sent and not sent_finish):
            out = model(input_ids=prev.unsqueeze(0), past_key_values=past, use_cache=True)
            logits = out.logits
            past = out.past_key_values

            if not topk:
                logits, indices = logits[0, -1, :].sort(descending=True)
            else:
                logits, indices = logits[0, -1, :151643].sort(descending=True)

            logits = logits.double()
            logits_temp = logits / temp
            probs_temp = F.softmax(logits_temp, dim=0)
            log_probs_temp = F.log_softmax(logits_temp, dim=0)
            log_probs = F.log_softmax(logits, dim=0)

            if i >= len(message):
                selection = 0
                sent_finish = is_sent_finish(indices[selection].item(), enc)
            else:
                cur_int_range = cur_interval[1] - cur_interval[0]
                cur_threshold = 1 / cur_int_range

                cutoff_indices = (probs_temp < cur_threshold).nonzero()
                if len(cutoff_indices) > 0:
                    k = max(2, cutoff_indices[0].item())
                else:
                    k = len(probs_temp)

                if topk:
                    k = min(k, topk)

                if not topk:
                    probs_temp_int = probs_temp[:k]
                else:
                    # Stepwise verification: remove tokens that cause tokenization inconsistency
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

                # Rescale to integer range and round
                probs_temp_int = probs_temp_int / probs_temp_int.sum() * cur_int_range
                probs_temp_int = probs_temp_int.round().long()
                cum_probs = probs_temp_int.cumsum(0)

                # Fix overfill from rounding
                overfill_index = (cum_probs > cur_int_range).nonzero()
                if len(overfill_index) > 0:
                    cum_probs = cum_probs[:overfill_index[0]]

                # Fix underfill
                cum_probs[-1] += cur_int_range - cum_probs[-1]

                probs_final = cum_probs.clone()
                probs_final[1:] = cum_probs[1:] - cum_probs[:-1]
                cum_probs += cur_interval[0]

                # Select token based on message bits
                message_bits = message[i:i + precision]
                if i + precision > len(message):
                    message_bits = message_bits + [0] * (i + precision - len(message))
                message_idx = bits2int(reversed(message_bits))
                selection = (cum_probs > message_idx).nonzero()[0].item()

                # Update interval
                new_int_bottom = cum_probs[selection - 1] if selection > 0 else cur_interval[0]
                new_int_top = cum_probs[selection]
                new_int_bottom_bits_inc = list(reversed(int2bits(new_int_bottom, precision)))
                new_int_top_bits_inc = list(reversed(int2bits(new_int_top - 1, precision)))
                num_bits_encoded = num_same_from_beg(new_int_bottom_bits_inc, new_int_top_bits_inc)
                i += num_bits_encoded

                new_int_bottom_bits = new_int_bottom_bits_inc[num_bits_encoded:] + [0] * num_bits_encoded
                new_int_top_bits = new_int_top_bits_inc[num_bits_encoded:] + [1] * num_bits_encoded
                cur_interval[0] = bits2int(reversed(new_int_bottom_bits))
                cur_interval[1] = bits2int(reversed(new_int_top_bits)) + 1

                # Gather statistics
                total_log_probs += log_probs[selection].item()
                q = probs_final.double() / probs_final.sum()
                logq = q.log()
                total_kl += kl(q, logq, log_probs[:len(q)])
                total_entropy_ptau += entropy(probs_temp, log_probs_temp)
                total_num_for_stats += 1

            prev = indices[selection].view(1)
            output = torch.cat((output, prev))
            num_bits += num_bits_encoded

            # Check for end-of-sequence in text mode
            partial = enc.tokenizer.decode(output[len(context):].tolist())
            if '<eos>' in partial:
                break

    avg_NLL = -total_log_probs / total_num_for_stats
    avg_KL = total_kl / total_num_for_stats
    avg_Hq = total_entropy_ptau / total_num_for_stats
    words_per_bit = total_num_for_stats / i

    out = output[len(context):].tolist()
    return out, avg_NLL, avg_KL, words_per_bit, avg_Hq


def decode_arithmetic(model, enc, text, context, device='cuda', temp=1.0, precision=16, topk=None):
    """Decode a bit-string from cover text tokens using arithmetic coding.

    Args:
        model: same autoregressive language model used for encoding
        enc: processor/tokenizer
        text: cover text (string or list of token ids)
        context: list of context token ids
        temp: must match the temperature used during encoding
        precision: must match the precision used during encoding
        topk: must match the topk used during encoding
    Returns:
        list of recovered bits
    """
    if isinstance(text, str):
        inp = enc.tokenizer.encode(text)
    elif isinstance(text, list):
        inp = torch.tensor(text, device=device, dtype=torch.long)
    else:
        inp = text

    max_val = 2 ** precision
    cur_interval = [0, max_val]
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

            if not topk:
                logits, indices = logits[0, -1, :].sort(descending=True)
            else:
                logits, indices = logits[0, -1, :151643].sort(descending=True)

            logits = logits.double()
            logits_temp = logits / temp
            probs_temp = F.softmax(logits_temp, dim=0)

            cur_int_range = cur_interval[1] - cur_interval[0]
            cur_threshold = 1 / cur_int_range

            cutoff_indices = (probs_temp < cur_threshold).nonzero()
            if len(cutoff_indices) > 0:
                k = max(2, cutoff_indices[0].item())
            else:
                k = len(probs_temp)

            if topk:
                k = min(k, topk)

            if not topk:
                probs_temp_int = probs_temp[:k]
            else:
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

            probs_temp_int = probs_temp_int / probs_temp_int.sum() * cur_int_range
            probs_temp_int = probs_temp_int.round().long()
            cum_probs = probs_temp_int.cumsum(0)

            overfill_index = (cum_probs > cur_int_range).nonzero()
            if len(overfill_index) > 0:
                cum_probs = cum_probs[:overfill_index[0]]
                k = overfill_index[0].item()

            cum_probs[-1] += cur_int_range - cum_probs[-1]
            cum_probs += cur_interval[0]

            rank = (indices == inp[i]).nonzero().item()
            if rank >= k:
                print('Error: tokenization inconsistency, rank >= k')
            selection = rank

            new_int_bottom = cum_probs[selection - 1] if selection > 0 else cur_interval[0]
            new_int_top = cum_probs[selection]
            new_int_bottom_bits_inc = list(reversed(int2bits(new_int_bottom, precision)))
            new_int_top_bits_inc = list(reversed(int2bits(new_int_top - 1, precision)))

            num_bits_encoded = num_same_from_beg(new_int_bottom_bits_inc, new_int_top_bits_inc)
            if i == len(inp) - 1:
                new_bits = new_int_bottom_bits_inc
            else:
                new_bits = new_int_top_bits_inc[:num_bits_encoded]
            message += new_bits

            new_int_bottom_bits = new_int_bottom_bits_inc[num_bits_encoded:] + [0] * num_bits_encoded
            new_int_top_bits = new_int_top_bits_inc[num_bits_encoded:] + [1] * num_bits_encoded
            cur_interval[0] = bits2int(reversed(new_int_bottom_bits))
            cur_interval[1] = bits2int(reversed(new_int_top_bits)) + 1

            prev = torch.tensor([indices[selection].item()], device=device, dtype=torch.long)
            num_bits += num_bits_encoded
            i += 1

    return message