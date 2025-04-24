import json, glob, statistics
from pathlib import Path
from utils import load_config, ensure_dataset, save_json


def aggregate():
    cfg = load_config()
    results_dir = Path(cfg['output_dir']) / 'per_strategy'

    # Extract all categories from the original dataset
    dataset = ensure_dataset(cfg)
    all_categories = sorted({sample.get('category', 'uncategorized') for sample in dataset})

    summary = {}
    # Iterate each strategy's log
    for log_file in sorted(results_dir.glob('*.json')):
        strat = log_file.stem
        data = json.load(open(log_file, 'r'))
        has_acc = any('accuracy' in rec for rec in data)

        # Prepare overall metrics
        overall_metrics = {
            'space_avg': [],
            'space_peak': [],
            'time_total': [],
            'time_per_token': [],
            'time_first_token': []
        }
        if has_acc:
            overall_metrics['accuracy'] = []

        # Populate overall lists
        for rec in data:
            overall_metrics['space_avg'].append(rec['sizes']['avg'])
            overall_metrics['space_peak'].append(rec['sizes']['peak'])
            overall_metrics['time_total'].append(rec['times']['total'])
            overall_metrics['time_per_token'].append(rec['times']['per_token'])
            overall_metrics['time_first_token'].append(rec['times']['first_token'])
            if has_acc:
                overall_metrics['accuracy'].append(rec.get('accuracy', 0))

        # Compute overall averages
        strat_summary = {'overall': {}, 'by_category': {}}
        for metric, vals in overall_metrics.items():
            strat_summary['overall'][metric] = statistics.mean(vals) if vals else 0.0

        # Per-category metrics
        for cat in all_categories:
            cat_metrics = {
                'space_avg': [],
                'space_peak': [],
                'time_total': [],
                'time_per_token': [],
                'time_first_token': []
            }
            if has_acc:
                cat_metrics['accuracy'] = []
            # Gather metrics for this category
            for rec in data:
                if rec.get('category', 'uncategorized') == cat:
                    cat_metrics['space_avg'].append(rec['sizes']['avg'])
                    cat_metrics['space_peak'].append(rec['sizes']['peak'])
                    cat_metrics['time_total'].append(rec['times']['total'])
                    cat_metrics['time_per_token'].append(rec['times']['per_token'])
                    cat_metrics['time_first_token'].append(rec['times']['first_token'])
                    if has_acc:
                        cat_metrics['accuracy'].append(rec.get('accuracy', 0))
            # Compute averages (0.0 if no entries)
            strat_summary['by_category'][cat] = {}
            for metric, vals in cat_metrics.items():
                strat_summary['by_category'][cat][metric] = statistics.mean(vals) if vals else 0.0

        summary[strat] = strat_summary

    # Save aggregated summary
    save_json(summary, cfg, '.', 'summary')
    print(f"Saved aggregated summary with per-category breakdown to {Path(cfg['output_dir'])/'summary.json'}")


if __name__ == '__main__':
    aggregate()
