import argparse
import json
from pathlib import Path

def update_config_with_new_strategies(config_path, output_path=None):
    """
    Update the configuration file to include the new strategies.
    
    Args:
        config_path: Path to the original config file
        output_path: Path to save the updated config file (if None, overwrites original)
    
    Returns:
        Path to the updated config file
    """
    # Load the original config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Add the new strategies
    config["strategies"] = [
        "Baseline",
        "Random(keep=0.7)",
        "AttentionTop(keep=0.7)",
        "AttentionBottom(keep=0.7)",
        "HybridNPercent(keep=0.7,r=0.5,a=0.3,t=0.2)",
        "SlidingWindow(window=0.7,important=0.1)",
        "AdaptiveAttention(base_keep=0.7)"
    ]
    
    # Save the updated config
    output_path = output_path or config_path
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    return output_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_template.json", help="Path to the original config file")
    parser.add_argument("--output", default=None, help="Path to save the updated config file (if None, overwrites original)")
    args = parser.parse_args()
    
    updated_path = update_config_with_new_strategies(args.config, args.output)
    print(f"Updated config saved to {updated_path}")

if __name__ == "__main__":
    main()
