"""
example_llama2_usage.py

Demonstrates how to use KVCacheManager with a Hugging Face Llama 2 7b model
to interactively manage the key-value cache during step-by-step inference.
"""
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer

from kv_cache_manager import KVCacheManager, WindowStrategy, NoOpStrategy

def main():
    # Model identifier for Llama 2 7B (base)
    model_name = "meta-llama/Llama-2-7b-hf"

    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer and model
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    # Ensure pad token is defined
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load model with caching enabled
    # For large models, consider using device_map='auto' and torch_dtype=torch.float16
    model = LlamaForCausalLM.from_pretrained(model_name)
    model.to(device)
    model.eval()

    # Choose a caching strategy: NoOp (keep all) or window (keep last N tokens)
    # strategy = NoOpStrategy()
    strategy = WindowStrategy(max_length=20)
    cache_manager = KVCacheManager(strategy)

    # Initial prompt
    prompt = "In a distant future, humanity has"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # First forward pass: build initial cache
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)
    past = cache_manager.update(outputs.past_key_values)
    print(f"After prompt '{prompt}':")
    for i, (k, _) in enumerate(past):
        print(f" Layer {i}: seq_len={k.size(-2)}")

    # Continue generation token by token
    continuation = " explorers have colonized dozens of planets"
    cont_ids = tokenizer(continuation, return_tensors="pt").input_ids.squeeze(0)
    generated_ids = inputs.input_ids

    for token_id in cont_ids:
        # single-token inference
        input_token = token_id.unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(input_ids=input_token, past_key_values=past, use_cache=True)
        past = cache_manager.update(outputs.past_key_values)
        generated_ids = torch.cat([generated_ids, input_token], dim=-1)
        decoded = tokenizer.decode(token_id.item(), skip_special_tokens=True)
        seq_len = past[0][0].size(-2)
        print(f"Generated '{decoded}' | new seq_len per layer: {seq_len}")

    # Final output
    final = tokenizer.decode(generated_ids.squeeze().tolist(), skip_special_tokens=True)
    print("\nFinal output:")
    print(final)

if __name__ == "__main__":
    main()