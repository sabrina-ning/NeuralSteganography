# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

'''
Repeats experiment logic from test_5.py, but using more recent/stronger language models
'''

num_tokens_to_generate = 10000
num_tokens_per_chunk = 2000
temperature = 0.9
top_k = 100

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("allenai/Olmo-3-1025-7B")

# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B-Base")
# model.to(device)
# model.eval()

# prompt_text = "Cornell University is a private Ivy League research university based in Ithaca, New York, United States. The university was co-founded by American philanthropist Ezra Cornell and historian and educator Andrew Dickson White in 1865."

# prompt_text = "Once upon a time, "

# prompt_text = """Mr. and Mrs. Dursley, of number four, Privet Drive, were proud to say that they were perfectly normal, thank you very much. They were the last people you'd expect to be involved in anything strange or mysterious, because they just didn't hold with such nonsense.\n
# Mr. Dursley was the director of a firm called Grunnings, which made drills. He was a big, beefy man with hardly any neck, although he did have a very large mustache. Mrs. Dursley was thin and blonde and had nearly twice the usual amount of neck, which came in very useful as she spent so much of her time craning over garden fences, spying on the neighbors. The Dursleys had a small son called Dudley and in their opinion there was no finer boy anywhere."""

# prompt_text = "Something in the structure of the world had begun to shift, though no one could agree on where the change started or what it meant. The first hints appeared not as events but as distortions in the stories people told about themselves."

# prompt_text = "He arrived in the city with only a rough plan and a set of assumptions he knew wouldn't survive first contact with reality. The streets were familiar in layout but different in feeling, as if the place had continued developing in his absence."

prompt_text = """The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attentionmechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 Englishto-German translation task, improving over the existing best results, including ensembles, by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data."""

inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
initial_input_ids = inputs["input_ids"]

print(f"Generating {num_tokens_to_generate} tokens")
print("\n--- Prompt ---")
print(prompt_text)

# print(f"\n\n--- Default Generation ---")
# # def model_generate(seed=1111):
# seed = 1111
# torch.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B-Base")
# model.to(device)
# model.eval()
# outputs = model.generate(
#     **inputs,
#     max_new_tokens=num_tokens_to_generate,
#     min_new_tokens=num_tokens_to_generate,
#     pad_token_id=tokenizer.eos_token_id,
#     do_sample=True,
#     temperature=temperature,
#     top_k=top_k,
#     top_p=1.0
# )
# # print(f"\n\n--- Default Generation (seed={seed}) ---")
# print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))

# reloads model per chunk
# print(f"\n\n--- Default Generation (version 2 - {num_tokens_per_chunk} token chunks) ---")
# seed = 1111
# torch.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)

# for _ in range(5):
#     for i in range(num_tokens_to_generate // num_tokens_per_chunk):
#         model = AutoModelForCausalLM.from_pretrained("allenai/Olmo-3-1025-7B")
#         model.to(device)
#         model.eval()

#         outputs = model.generate(
#             **inputs,
#             max_new_tokens=num_tokens_per_chunk,
#             min_new_tokens=num_tokens_per_chunk,
#             # pad_token_id=tokenizer.eos_token_id,
#             do_sample=True,
#             temperature=temperature,
#             top_k=top_k,
#             top_p=1.0
#         )
#         output_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:])
#         print(output_text)
#         inputs = tokenizer(output_text, return_tensors="pt").to(device)

#     # breakpoint()
#     print("MARKER")


print("\n--- Manual Generation ---")
# seed = 1111
# torch.manual_seed(seed)
# torch.cuda.manual_seed_all(seed) 

model = AutoModelForCausalLM.from_pretrained("allenai/Olmo-3-1025-7B")
model.to(device)
model.eval()

generated_token_ids = []
current_ids = initial_input_ids
past_key_values = None

num_post_truncation = 0

with torch.no_grad():
    outputs = model(current_ids, use_cache=True)
    past_key_values = outputs.past_key_values
    next_token_logits = outputs.logits[:, -1, :].to(dtype=torch.float32)
    
    for i in range(num_tokens_to_generate):
        # FIXED ORDER: top-k filtering first
        v, _ = torch.topk(next_token_logits, top_k)
        kth_value = v[:, -1].unsqueeze(-1)
        filtered_logits = torch.where(
            next_token_logits < kth_value, 
            -float('inf'), 
            next_token_logits
        )
        
        # then apply temperature
        scaled_logits = filtered_logits / temperature

        if i < num_tokens_to_generate - 1:
             scaled_logits[:, tokenizer.eos_token_id] = -float('inf') # disallow eos token
        
        probabilities = torch.nn.functional.softmax(scaled_logits, dim=-1)
        
        next_token_id = torch.multinomial(probabilities, num_samples=1)
        
        generated_token_ids.append(next_token_id.item())
        num_post_truncation += 1

        if num_post_truncation == num_tokens_per_chunk:
            truncated_input = torch.tensor([generated_token_ids[-num_tokens_per_chunk:]], dtype=torch.long).to(model.device)
            output_text = tokenizer.decode(generated_token_ids)
            print(output_text)
            inputs = tokenizer(output_text, return_tensors="pt").to(device)
            print("MARKER")

            model = AutoModelForCausalLM.from_pretrained("allenai/Olmo-3-1025-7B")
            model.to(device)
            model.eval()

            outputs = model(truncated_input, use_cache=True)
            # outputs = model(inputs, use_cache=True)
            num_post_truncation = 0
            generated_token_ids = []
        else:
            outputs = model(
                input_ids=next_token_id, 
                use_cache=True, 
                past_key_values=past_key_values
            )
        
        past_key_values = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :].to(dtype=torch.float32)

print(tokenizer.decode(generated_token_ids, skip_special_tokens=True))
