#!/usr/bin/env python3
"""
run_llama3_experiment.py

Wrapper script to run KV cache management benchmarks for the Llama3 model
using the existing robust_real_benchmark.py machinery and MTBench dataset.
This does not modify existing code; it generates a temporary config file
with the model name overridden to a Llama3 checkpoint.
"""
import argparse
import json
import os
import sys
import subprocess

def main():
    parser = argparse.ArgumentParser(
        description="Run KV cache management experiments for Llama3 model"
    )
    parser.add_argument(
        '--base_config',
        type=str,
        default='config.json',
        help='Path to the base config file (with Llama2 settings)'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='meta-llama/Meta-Llama-3-8b-instruct',
        help='HuggingFace model name for the Llama3 checkpoint'
    )
    parser.add_argument(
        '--limit',
        type=int,
        help='Limit number of samples to process'
    )
    parser.add_argument(
        '--eval_azure',
        action='store_true',
        help='Enable Azure OpenAI evaluation'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    parser.add_argument(
        '--profile',
        action='store_true',
        help='Enable memory profiling'
    )
    parser.add_argument(
        '--generate_dashboard',
        action='store_true',
        help='Generate a dashboard after benchmarking'
    )
    args, extra = parser.parse_known_args()

    # Load and override config
    with open(args.base_config, 'r') as f:
        cfg = json.load(f)
    # Override model name for Llama3
    cfg['model_name'] = args.model_name
    # Ensure multi-turn dataset prompt_field for MTBench
    if isinstance(cfg.get('dataset'), dict) and cfg['dataset'].get('name') == 'mtbench':
        cfg['dataset']['prompt_field'] = 'turns'

    # Write overridden config to a new file alongside this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    llama3_config = os.path.join(script_dir, 'config_llama3.json')
    with open(llama3_config, 'w') as f:
        json.dump(cfg, f, indent=2)
    print(f"Generated Llama3 config at {llama3_config}")

    # Build command to invoke the benchmark script
    benchmark_script = os.path.join(script_dir, 'robust_real_benchmark.py')
    cmd = [sys.executable, benchmark_script, '--config', llama3_config]
    if args.eval_azure:
        cmd.append('--eval_azure')
    if args.debug:
        cmd.append('--debug')
    if args.profile:
        cmd.append('--profile')
    if args.limit is not None:
        cmd.extend(['--limit', str(args.limit)])
    if args.generate_dashboard:
        cmd.append('--generate_dashboard')
    # Forward any extra args
    cmd.extend(extra)

    # Execute benchmarking
    result = subprocess.run(cmd)
    sys.exit(result.returncode)

if __name__ == '__main__':
    main()