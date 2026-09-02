"""Image-mode arithmetic coding for Emu3 image steganography.

Extends the text-mode arithmetic coder to handle Emu3's image token generation,
where the token sequence has a fixed spatial structure:
  - Visual tokens arranged in rows of `width` tokens
  - End-of-line (EOL) tokens after each row (deterministic)
  - End-of-frame (EOF), end-of-image (EOI), end-of-sequence (EOS) tokens at the end

Deterministic structural tokens are emitted without consuming bits from the message.
Only the visual tokens (which have a learned probability distribution) carry hidden information.
"""

import torch
import torch.nn.functional as F
from transformers import DynamicCache
import random
import time

from src.utils import limit_past, kl, entropy, bits2int, int2bits, is_sent_finish, num_same_from_beg, is_cit


def get_gen_mask(input_ids, logits, width, height):
    """Returns a mask to constrain image generation.

    Based on the current position in the image grid, returns a logit mask
    that forces deterministic structural tokens (EOL, EOF, EOI, EOS) at
    the correct positions, and allows visual tokens elsewhere.

    Returns:
        (mask, is_deterministic): mask tensor and whether this position is deterministic
    """
    # Emu3 special token IDs
    image_wrapper_token_id = 151851
    eol_token_id = 151846
    eof_token_id = 151847
    eoi_token_id = 151853
    eos_token_id = 151850
    visual_start_id = 151854

    mask = torch.full_like(logits, -float('inf'), device="cuda")

    # Find start of image
    image_wrappers = (input_ids == image_wrapper_token_id).nonzero(as_tuple=True)
    if len(image_wrappers[1]) == 0:
        mask[image_wrapper_token_id] = 0.0
        return mask, True

    offset = input_ids.shape[1] - image_wrappers[1][-1]
    row_length = width + 1
    is_deterministic = True

    if offset % row_length == 0:
        mask[eol_token_id] = 0.0
    elif offset == row_length * height + 1:
        mask[eof_token_id] = 0.0
    elif offset == row_length * height + 2:
        mask[eoi_token_id] = 0.0
    elif offset == row_length * height + 3:
        mask[eos_token_id] = 0.0
    elif offset > row_length * height + 3:
        mask[image_wrapper_token_id] = 0.0
    else:
        mask[visual_start_id:] = 0.0
        is_deterministic = False

    return mask, is_deterministic


def encode_arithmetic(model, enc, message, context, width=90, height=90,
                      finish_sent=False, device='cuda', temp=1.0, precision=16, topk=None):
    """Encode bits into Emu3 image tokens using arithmetic coding.

    The encoder generates image tokens autoregressively, selecting each visual
    token to encode bits from the message. Structural tokens (EOL, EOF, EOI, EOS)
    are emitted deterministically without consuming bits.

    Supports multi-image encoding: if the message doesn't fit in one image,
    generation continues with a new image.

    Args:
        model: Emu3ForConditionalGeneration model
        enc: Emu3 processor
        message: list of bits to encode
        context: context token tensor or list
        width, height: image dimensions in token-grid units (pixels / spatial_factor)
        temp: sampling temperature
        precision: arithmetic coder precision in bits
        topk: if set, restrict to top-k visual tokens
    Returns:
        (tokens, avg_NLL, avg_KL, words_per_bit, avg_Hq)
    """
    start_time = time.perf_counter()
    total_num_bits = len(message)

    if isinstance(context, list):
        context = torch.tensor(context, device=device, dtype=torch.long)

    eos_token_id = 151850
    max_val = 2 ** precision
    cur_interval = [0, max_val]

    prev = context
    output = context
    past = None
    acc_output = torch.tensor([], device=device, dtype=torch.long)

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

            # Deterministic tokens (EOL/EOF/EOI/EOS) don't consume bits
            if is_deterministic:
                selection = torch.argmax(masked_logits).item()
                prev = torch.tensor([selection], device=device)
                output = torch.cat((output, prev))

                if selection == eos_token_id:
                    acc_output = torch.cat((acc_output, output))
                    if num_bits >= total_num_bits:
                        print(f"\nAll {num_bits} bits encoded")
                        break
                    else:
                        print(f"\nImage complete, but ({num_bits}/{total_num_bits}) bits remain. Starting new image...")
                        prev = context
                        output = context
                        past = None
                continue

            logits_temp = masked_logits / temp
            probs_temp = F.softmax(logits_temp, dim=0)
            probs_sorted, indices_sorted = torch.sort(probs_temp, descending=True)

            cur_int_range = cur_interval[1] - cur_interval[0]
            cur_threshold = 1.0 / cur_int_range

            cutoff_indices = (probs_sorted < cur_threshold).nonzero()
            if len(cutoff_indices) > 0:
                k = max(2, cutoff_indices[0].item())
            else:
                k = len(probs_sorted)
            if topk:
                k = min(k, topk)

            probs_temp_int = probs_sorted[:k]
            indices = indices_sorted[:k]

            probs_temp_int = probs_temp_int / probs_temp_int.sum() * cur_int_range
            probs_temp_int = probs_temp_int.round().long()
            cum_probs = probs_temp_int.cumsum(0)

            overfill_index = (cum_probs > cur_int_range).nonzero()
            if len(overfill_index) > 0:
                cum_probs = cum_probs[:overfill_index[0]]
                indices = indices[:len(cum_probs)]

            cum_probs[-1] += cur_int_range - cum_probs[-1]

            probs_final = cum_probs.clone()
            probs_final[1:] = cum_probs[1:] - cum_probs[:-1]
            cum_probs += cur_interval[0]

            # Pad message with random bits if needed
            if i + precision > len(message):
                padding = [random.randint(0, 1) for _ in range(i + precision - len(message))]
                message.extend(padding)

            message_bits = message[i:i + precision]
            message_idx = bits2int(reversed(message_bits))
            selection_idx = (cum_probs > message_idx).nonzero()[0].item()

            # Update interval
            new_int_bottom = cum_probs[selection_idx - 1] if selection_idx > 0 else cur_interval[0]
            new_int_top = cum_probs[selection_idx]
            new_int_bottom_bits_inc = list(reversed(int2bits(new_int_bottom, precision)))
            new_int_top_bits_inc = list(reversed(int2bits(new_int_top - 1, precision)))
            num_bits_encoded = num_same_from_beg(new_int_bottom_bits_inc, new_int_top_bits_inc)
            i += num_bits_encoded

            new_int_bottom_bits = new_int_bottom_bits_inc[num_bits_encoded:] + [0] * num_bits_encoded
            new_int_top_bits = new_int_top_bits_inc[num_bits_encoded:] + [1] * num_bits_encoded
            cur_interval[0] = bits2int(reversed(new_int_bottom_bits))
            cur_interval[1] = bits2int(reversed(new_int_top_bits)) + 1

            total_num_for_stats += 1

            prev = indices[selection_idx].view(1)
            output = torch.cat((output, prev))
            num_bits += num_bits_encoded
            print(f"\rEncoded {num_bits}/{total_num_bits} bits", end="")

    out = acc_output
    print(f"\nTotal bits encoded: {num_bits}")
    print(f"Encoding took {time.perf_counter() - start_time:.1f} seconds")
    return out, -total_log_probs / max(1, total_num_for_stats), total_kl / max(1, total_num_for_stats), total_num_for_stats / max(1, i), 0


def decode_arithmetic(model, enc, text, context, width=90, height=90,
                      device='cuda', temp=1.0, precision=16, topk=None):
    """Decode bits from Emu3 image tokens using arithmetic coding.

    The decoder replays the same generation process used during encoding,
    using the known token sequence to recover the original message bits.

    Args:
        model: same Emu3 model used for encoding
        enc: Emu3 processor
        text: list of image tokens (or tensor)
        context: context token tensor or list
        width, height: must match encoding dimensions
        temp: must match encoding temperature
        precision: must match encoding precision
        topk: must match encoding topk
    Returns:
        list of recovered bits
    """
    start_time = time.perf_counter()
    if isinstance(text, list):
        inp = torch.tensor(text, device=device, dtype=torch.long)
    else:
        inp = text

    if isinstance(context, list):
        context = torch.tensor(context, device=device, dtype=torch.long)

    eos_token_id = 151850
    max_val = 2 ** precision
    cur_interval = [0, max_val]

    prev = context
    full_seq = context
    past = None
    message = []

    with torch.no_grad():
        i = 0
        while i < len(inp):
            target_token = inp[i]

            out = model(input_ids=prev.unsqueeze(0), past_key_values=past, use_cache=True)
            logits = out.logits
            past = out.past_key_values

            next_token_logits = logits[0, -1, :].to(dtype=torch.float32)
            mask, is_deterministic = get_gen_mask(full_seq.unsqueeze(0), next_token_logits, width, height)
            masked_logits = next_token_logits + mask

            if is_deterministic:
                prev = target_token.view(1)
                full_seq = torch.cat((full_seq, prev))
                if target_token == eos_token_id:
                    print("Finished decoding one image.")
                i += 1
                continue

            logits_temp = masked_logits / temp
            probs_temp = F.softmax(logits_temp, dim=0)
            probs_sorted, indices_sorted = torch.sort(probs_temp, descending=True)

            cur_int_range = cur_interval[1] - cur_interval[0]
            cur_threshold = 1.0 / cur_int_range

            cutoff_indices = (probs_sorted < cur_threshold).nonzero()
            if len(cutoff_indices) > 0:
                k = max(2, cutoff_indices[0].item())
            else:
                k = len(probs_sorted)
            if topk:
                k = min(k, topk)

            probs_temp_int = probs_sorted[:k]
            indices = indices_sorted[:k]

            probs_temp_int = probs_temp_int / probs_temp_int.sum() * cur_int_range
            probs_temp_int = probs_temp_int.round().long()
            cum_probs = probs_temp_int.cumsum(0)

            overfill_index = (cum_probs > cur_int_range).nonzero()
            if len(overfill_index) > 0:
                cum_probs = cum_probs[:overfill_index[0]]
                indices = indices[:len(cum_probs)]

            cum_probs[-1] += cur_int_range - cum_probs[-1]
            cum_probs += cur_interval[0]

            # Find the target token's position
            rank_matches = (indices == target_token).nonzero()
            if len(rank_matches) == 0:
                print(f"\nWarning: Target token {target_token} not in top-{k} at index {i}")
                print("Decoded message may be corrupted from this point.")
                break

            selection_idx = rank_matches[0].item()
            new_int_bottom = cum_probs[selection_idx - 1] if selection_idx > 0 else cur_interval[0]
            new_int_top = cum_probs[selection_idx]

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

            prev = target_token.view(1)
            full_seq = torch.cat((full_seq, prev))
            print(f"\rDecoded {len(message)} bits...", end="")
            i += 1

    print(f"\nDecoding took {time.perf_counter() - start_time:.1f} seconds")
    return message
