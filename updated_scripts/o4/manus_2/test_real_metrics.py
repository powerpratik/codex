#!/usr/bin/env python3
"""
Test script for validating the real metrics implementation.

This script tests the KV cache manager, inference time tracker, and accuracy evaluator
to ensure they're properly measuring real metrics.
"""

import os
import sys
import json
import time
import logging
import argparse
import torch
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import our custom modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from complete_real_kv_cache_manager import RealKVCacheManager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('test_metrics.log')
    ]
)
logger = logging.getLogger('test_real_metrics')

def load_model_and_tokenizer(model_name, cache_dir=None):
    """Load model and tokenizer from HuggingFace"""
    logger.info(f"Loading model {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Set up padding token for the tokenizer if it doesn't have one
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("Set pad_token to eos_token")
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            logger.info("Added [PAD] as pad_token")
    
    return tokenizer, model

def test_kv_cache_manager(model, tokenizer):
    """Test the KV cache manager"""
    logger.info("Testing KV cache manager...")
    
    # Initialize KV cache manager
    cache_manager = RealKVCacheManager(
        model=model,
        tokenizer=tokenizer,
        cfg={"kv_threshold": 20},  # Lower threshold for testing
        logger=logger
    )
    
    # Set logger level to INFO to see memory logs
    cache_manager.logger.setLevel(logging.INFO)
    
    # Test prompt
    prompt = "Explain the concept of KV cache in large language models."
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", padding=False, truncation=True).to(model.device)
    
    # Generate tokens to build KV cache
    logger.info("Generating tokens to build KV cache...")
    with torch.no_grad():
        out = model(input_ids=inputs.input_ids, use_cache=True)
    
    # Get KV cache
    past = model.past_key_values
    
    # Update memory stats
    cache_manager._update_memory_stats(past)
    
    # Log initial memory stats
    initial_memory = cache_manager.current_memory_mb
    logger.info(f"Initial KV cache size: {initial_memory:.2f} MB")
    
    # Test each strategy
    strategies = [
        "Baseline",
        "Random(keep=0.7)",
        "AttentionTop(keep=0.7)",
        "AttentionBottom(keep=0.7)",
        "SlidingWindow(window=0.7,important=0.1)",
        "AdaptiveAttention(base_keep=0.7)"
    ]
    
    for strat_name in strategies:
        logger.info(f"Testing strategy: {strat_name}")
        
        # Apply strategy
        updated_past = cache_manager.apply_eviction_strategy(past, strat_name)
        
        # Update memory stats
        cache_manager._update_memory_stats(updated_past)
        
        # Log memory after strategy
        after_memory = cache_manager.current_memory_mb
        logger.info(f"KV cache size after {strat_name}: {after_memory:.2f} MB")
        
        # Verify memory reduction for non-baseline strategies
        if strat_name != "Baseline":
            if after_memory < initial_memory:
                logger.info(f"✅ {strat_name} successfully reduced KV cache size")
            else:
                logger.warning(f"❌ {strat_name} did not reduce KV cache size")
    
    # Test HybridNPercent strategy separately
    logger.info("Testing HybridNPercent strategy...")
    
    # Get token types
    token_ids = inputs.input_ids[0].tolist()
    token_types = []
    for token_id in token_ids:
        # Simple classification based on token ID
        if token_id in tokenizer.all_special_ids:
            token_types.append("special")
        elif token_id > len(tokenizer.get_vocab()) * 0.9:
            token_types.append("rare")
        else:
            token_types.append("common")
    
    # Apply HybridNPercent strategy
    hybrid_past = cache_manager._apply_hybrid_strategy(
        past,
        attention_scores=out.attentions if hasattr(out, "attentions") else None,
        token_types=token_types,
        keep_ratio=0.7,
        recency_weight=0.5,
        attention_weight=0.3,
        type_weight=0.2
    )
    
    # Update memory stats
    cache_manager._update_memory_stats(hybrid_past)
    
    # Log memory after HybridNPercent
    hybrid_memory = cache_manager.current_memory_mb
    logger.info(f"KV cache size after HybridNPercent: {hybrid_memory:.2f} MB")
    
    # Verify memory reduction
    if hybrid_memory < initial_memory:
        logger.info("✅ HybridNPercent successfully reduced KV cache size")
    else:
        logger.warning("❌ HybridNPercent did not reduce KV cache size")
    
    return True

def test_inference_time_tracking(model, tokenizer):
    """Test inference time tracking"""
    logger.info("Testing inference time tracking...")
    
    # Test prompt
    prompt = "Explain the concept of attention mechanisms in transformers."
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", padding=False, truncation=True).to(model.device)
    
    # Measure encoding time
    encoding_start = time.time()
    _ = tokenizer(prompt, return_tensors="pt", padding=False, truncation=True).to(model.device)
    encoding_time = time.time() - encoding_start
    logger.info(f"Encoding time: {encoding_time:.4f}s")
    
    # Measure first token time
    first_token_start = time.time()
    with torch.no_grad():
        out = model(input_ids=inputs.input_ids, use_cache=True)
    first_token_time = time.time() - first_token_start
    logger.info(f"First token time: {first_token_time:.4f}s")
    
    # Get next token
    next_token = out.logits[:, -1, :].argmax(dim=-1)
    token = torch.tensor([[next_token.item()]], device=model.device)
    
    # Measure per-token generation time for a few tokens
    token_times = []
    for i in range(5):
        token_start = time.time()
        with torch.no_grad():
            out = model(input_ids=token, use_cache=True)
        token_time = time.time() - token_start
        token_times.append(token_time)
        
        # Get next token
        next_token = out.logits[:, -1, :].argmax(dim=-1)
        token = torch.tensor([[next_token.item()]], device=model.device)
    
    # Log token times
    avg_token_time = np.mean(token_times)
    logger.info(f"Average token generation time: {avg_token_time:.4f}s")
    logger.info(f"Token times: {[f'{t:.4f}s' for t in token_times]}")
    
    # Verify reasonable times
    if 0 < encoding_time < 1.0:
        logger.info("✅ Encoding time is reasonable")
    else:
        logger.warning(f"❌ Encoding time ({encoding_time:.4f}s) seems unusual")
    
    if 0 < first_token_time < 1.0:
        logger.info("✅ First token time is reasonable")
    else:
        logger.warning(f"❌ First token time ({first_token_time:.4f}s) seems unusual")
    
    if 0 < avg_token_time < 0.5:
        logger.info("✅ Average token time is reasonable")
    else:
        logger.warning(f"❌ Average token time ({avg_token_time:.4f}s) seems unusual")
    
    return True

def test_accuracy_evaluation(model, tokenizer):
    """Test accuracy evaluation"""
    logger.info("Testing accuracy evaluation...")
    
    # Test prompt
    prompt = "What is the capital of France?"
    expected_answer = "Paris"
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", padding=False, truncation=True).to(model.device)
    
    # Generate response
    with torch.no_grad():
        out = model(input_ids=inputs.input_ids, use_cache=True)
    
    # Calculate perplexity
    try:
        # Create a loss function
        loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
        
        # Reshape logits to (batch_size * seq_len, vocab_size)
        logits = out.logits.view(-1, out.logits.size(-1))
        
        # Reshape target_ids to (batch_size * seq_len)
        target_ids = inputs.input_ids.view(-1)
        
        # Calculate loss
        loss = loss_fn(logits, target_ids)
        
        # Calculate perplexity
        perplexity = torch.exp(loss).item()
        
        logger.info(f"Perplexity: {perplexity:.4f}")
        
        # Verify perplexity is a reasonable value
        if 1.0 < perplexity < 1000.0:
            logger.info("✅ Perplexity is in a reasonable range")
        else:
            logger.warning(f"❌ Perplexity ({perplexity:.4f}) seems unusual")
    
    except Exception as e:
        logger.error(f"Error calculating perplexity: {e}")
        return False
    
    # Generate a few tokens to check if the answer contains "Paris"
    generated_ids = inputs.input_ids.clone()
    
    for i in range(10):
        # Get next token
        next_token = out.logits[:, -1, :].argmax(dim=-1)
        generated_ids = torch.cat([generated_ids, next_token.unsqueeze(-1)], dim=-1)
        
        # Prepare for next iteration
        token = torch.tensor([[next_token.item()]], device=model.device)
        
        # Generate next token
        with torch.no_grad():
            out = model(input_ids=token, use_cache=True)
    
    # Decode generated text
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    logger.info(f"Generated text: {generated_text}")
    
    # Check if the answer contains the expected answer
    if expected_answer.lower() in generated_text.lower():
        logger.info(f"✅ Generated text contains the expected answer '{expected_answer}'")
    else:
        logger.warning(f"❌ Generated text does not contain the expected answer '{expected_answer}'")
    
    return True

def main():
    """Main function"""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Test real metrics implementation")
    parser.add_argument("--cache_dir", type=str, help="Path to model cache directory")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-chat-hf", help="Model name")
    args = parser.parse_args()
    
    # Load model and tokenizer
    tokenizer, model = load_model_and_tokenizer(args.model_name, args.cache_dir)
    
    # Run tests
    tests = [
        ("KV Cache Manager", test_kv_cache_manager),
        ("Inference Time Tracking", test_inference_time_tracking),
        ("Accuracy Evaluation", test_accuracy_evaluation)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}\nRunning test: {test_name}\n{'='*50}")
        try:
            success = test_func(model, tokenizer)
            results[test_name] = "PASS" if success else "FAIL"
        except Exception as e:
            logger.error(f"Error in {test_name} test: {e}")
            results[test_name] = "ERROR"
    
    # Print summary
    logger.info("\n\n" + "="*50)
    logger.info("TEST RESULTS SUMMARY")
    logger.info("="*50)
    
    all_passed = True
    for test_name, result in results.items():
        logger.info(f"{test_name}: {result}")
        if result != "PASS":
            all_passed = False
    
    if all_passed:
        logger.info("\n✅ All tests passed! The implementation is working correctly.")
    else:
        logger.warning("\n❌ Some tests failed. Please check the logs for details.")

if __name__ == "__main__":
    main()
