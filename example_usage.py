"""
example_usage.py

Demonstrates how to use KVCacheManager with a Hugging Face GPT-2 model
to interactively manage the key-value cache during step-by-step inference.
"""
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from kv_cache_manager import KVCacheManager, WindowStrategy, NoOpStrategy

def main():
    # Load model and tokenizer
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()

    # Choose a caching strategy: NoOp (keep all) or window (keep last N tokens)
    #strategy = NoOpStrategy()
    strategy = WindowStrategy(max_length=5)
    cache_manager = KVCacheManager(strategy)

    # Initial prompt
    prompt = "The quick brown fox"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    # First forward pass: no cache
    with torch.no_grad():
        outputs = model(input_ids=input_ids, use_cache=True)
    past = cache_manager.update(outputs.past_key_values)
    print(f"After prompt '{prompt}':")
    for layer_id, (k, v) in enumerate(past):
        print(f" Layer {layer_id}: seq_len={k.size(-2)}")

    # Generate step-by-step for a continuation
    continuation = " jumps over the lazy dog"
    continuation_ids = tokenizer(continuation, return_tensors="pt").input_ids[0]
    generated = input_ids

    for token_id in continuation_ids:
        # Prepare single-token input
        input_token = torch.tensor([[token_id]])
        with torch.no_grad():
            outputs = model(input_ids=input_token, past_key_values=past, use_cache=True)
        past = cache_manager.update(outputs.past_key_values)
        generated = torch.cat((generated, input_token), dim=1)
        decoded = tokenizer.decode(input_token.squeeze())
        print(f"Generated: '{decoded}' | new seq_len per layer: {past[0][0].size(-2)}")

    # Final decoded output
    print("Final output:", tokenizer.decode(generated.squeeze()))

if __name__ == "__main__":
    main()