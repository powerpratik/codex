import argparse
import json
import logging
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

from real_kv_cache_manager import RealKVCacheManager
from real_inference_time_tracker import InferenceTimeTracker
from real_accuracy_evaluator import AccuracyEvaluator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("test_metrics.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("test_metrics")

def load_model_and_tokenizer(model_name, cache_dir):
    """Load model and tokenizer with proper configuration"""
    logger.info(f"Loading tokenizer for {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    
    logger.info(f"Loading model {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        device_map="auto", 
        torch_dtype="auto",
        cache_dir=cache_dir
    )
    
    # Print model device information
    devices = set()
    for param in model.parameters():
        devices.add(param.device)
    logger.info(f"Model loaded on devices: {devices}")
    
    return tokenizer, model

def test_kv_cache_measurement(model, tokenizer, prompt):
    """Test KV cache size measurement"""
    logger.info("Testing KV cache size measurement...")
    
    # Create config
    config = {"kv_threshold": 1000}
    
    # Create KV cache manager
    cache_manager = RealKVCacheManager(model, tokenizer, config, logger)
    
    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate with KV cache tracking
    with torch.no_grad():
        # Initial encoding
        outputs = model(**inputs, use_cache=True, output_attentions=True)
        past = outputs.past_key_values
        
        # Measure initial cache size
        initial_size = cache_manager.compute_cache_size(past)
        logger.info(f"Initial KV cache size: {initial_size:.2f} MB")
        
        # Generate a few tokens to see cache growth
        token = inputs.input_ids[:, -1:].to(model.device)
        sizes = [initial_size]
        
        for i in range(10):
            outputs = model(
                input_ids=token, 
                past_key_values=past,
                use_cache=True, 
                output_attentions=True
            )
            
            # Get next token
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            
            # Update KV cache
            past = outputs.past_key_values
            
            # Measure cache size
            size = cache_manager.compute_cache_size(past)
            sizes.append(size)
            
            # Prepare for next iteration
            token = next_token
        
        # Apply eviction strategy
        logger.info("Testing eviction strategy...")
        strategy = "SlidingWindow"
        past = cache_manager.apply_eviction_strategy(past, strategy)
        
        # Measure post-eviction size
        post_eviction_size = cache_manager.compute_cache_size(past)
        
        # Clean up
        cache_manager.remove_hooks()
        
        return {
            "initial_size_mb": initial_size,
            "final_size_mb": sizes[-1],
            "size_growth": sizes,
            "post_eviction_size_mb": post_eviction_size,
            "eviction_reduction_percent": (1 - post_eviction_size / sizes[-1]) * 100
        }

def test_inference_time_measurement(model, tokenizer, prompt):
    """Test inference time measurement"""
    logger.info("Testing inference time measurement...")
    
    # Create time tracker
    time_tracker = InferenceTimeTracker(model, tokenizer, logger)
    
    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_tokens = inputs.input_ids[0].tolist()
    
    # Start encoding with timing
    time_tracker.start_encoding()
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)
    time_tracker.end_encoding(input_tokens)
    
    # Get the cache object from the output
    past = outputs.past_key_values
    
    # Prepare for generation
    token = inputs.input_ids[:, -1:].to(model.device)
    gen_tokens = input_tokens.copy()
    
    # Start generation with timing
    time_tracker.start_generation()
    
    # Generate a few tokens
    for i in range(10):
        with torch.no_grad():
            outputs = model(
                input_ids=token, 
                past_key_values=past,
                use_cache=True
            )
        
        # Get next token
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        next_token_id = next_token.item()
        gen_tokens.append(next_token_id)
        
        # Record token generation time
        time_tracker.token_generated(next_token_id)
        
        # Update KV cache
        past = outputs.past_key_values
        
        # Prepare for next iteration
        token = next_token
    
    # End generation
    time_tracker.end_generation()
    
    # Get time stats
    time_stats = time_tracker.get_time_stats()
    
    # Clean up
    time_tracker.remove_hooks()
    
    return {
        "encoding_time_sec": time_stats["encoding_time"],
        "total_generation_time_sec": time_stats["total_generation_time"],
        "first_token_time_sec": time_stats["first_token_time"],
        "avg_token_time_sec": time_stats["avg_token_time"],
        "tokens_per_second": time_stats["tokens_per_second"],
        "token_times": time_stats["token_times"]
    }

def test_accuracy_measurement(model, tokenizer, prompt):
    """Test accuracy measurement"""
    logger.info("Testing accuracy measurement...")
    
    # Create config
    config = {"use_azure": False}  # Disable Azure for testing
    
    # Create accuracy evaluator
    accuracy_evaluator = AccuracyEvaluator(model, tokenizer, config, logger)
    
    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_tokens = inputs.input_ids[0].tolist()
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=20,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True
        )
    
    # Get generated tokens
    generated_ids = outputs.sequences[0].tolist()
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    # Evaluate response
    metrics = accuracy_evaluator.evaluate_response(
        prompt, generated_text, input_tokens, generated_ids
    )
    
    return {
        "perplexity": metrics["perplexity"],
        "generated_text": generated_text
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--cache_dir", default=None)
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path("test_results")
    output_dir.mkdir(exist_ok=True)
    
    # Load model and tokenizer
    tokenizer, model = load_model_and_tokenizer(args.model_name, args.cache_dir)
    
    # Set model to evaluation mode
    model.eval()
    
    # Test prompt
    prompt = "Explain the concept of KV cache in large language models and why it's important for inference efficiency."
    
    # Run tests
    results = {}
    
    # Test KV cache measurement
    logger.info("Running KV cache measurement test...")
    kv_results = test_kv_cache_measurement(model, tokenizer, prompt)
    results["kv_cache"] = kv_results
    
    # Test inference time measurement
    logger.info("Running inference time measurement test...")
    time_results = test_inference_time_measurement(model, tokenizer, prompt)
    results["inference_time"] = time_results
    
    # Test accuracy measurement
    logger.info("Running accuracy measurement test...")
    accuracy_results = test_accuracy_measurement(model, tokenizer, prompt)
    results["accuracy"] = accuracy_results
    
    # Save results
    with open(output_dir / "test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    logger.info("\n=== Test Results Summary ===")
    logger.info(f"KV Cache Initial Size: {kv_results['initial_size_mb']:.2f} MB")
    logger.info(f"KV Cache Final Size: {kv_results['final_size_mb']:.2f} MB")
    logger.info(f"KV Cache Eviction Reduction: {kv_results['eviction_reduction_percent']:.2f}%")
    logger.info(f"Encoding Time: {time_results['encoding_time_sec']:.4f} sec")
    logger.info(f"First Token Time: {time_results['first_token_time_sec']:.4f} sec")
    logger.info(f"Average Token Time: {time_results['avg_token_time_sec']:.4f} sec")
    logger.info(f"Tokens Per Second: {time_results['tokens_per_second']:.2f}")
    logger.info(f"Perplexity: {accuracy_results['perplexity']:.2f}")
    logger.info("Test results saved to test_results/test_results.json")

if __name__ == "__main__":
    main()
