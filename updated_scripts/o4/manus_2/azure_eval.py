import json
import argparse
import time
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np
from openai import AzureOpenAI
from utils import load_config

class LLMResponseEvaluator:
    """Evaluates LLM responses using Azure OpenAI as a judge."""
    def __init__(self,
                 api_key: str,
                 endpoint: str,
                 api_version: str,
                 deployment_name: str,
                 max_workers: int = 5,
                 retry_count: int = 3,
                 retry_delay: int = 2,
                 skip_filtered: bool = False):
        self.client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=endpoint
        )
        self.deployment_name = deployment_name
        self.max_workers = max_workers
        self.retry_count = retry_count
        self.retry_delay = retry_delay
        self.skip_filtered = skip_filtered
        self.categories = set()
        self.skipped_responses = []

    def load_data(self, input_file: str) -> dict:
        with open(input_file, 'r') as f:
            data = json.load(f)
        strategies = {}
        for item in data:
            strat = item.get('strategy', 'unknown')
            strategies.setdefault(strat, [])
            for resp in item.get('responses', []):
                resp['strategy'] = strat
                cat = resp.get('category', 'uncategorized')
                self.categories.add(cat)
                strategies[strat].append(resp)
        return strategies

    def evaluate_response(self, response_data: dict) -> dict:
        prompt = response_data['prompt']
        response = response_data['response']
        category = response_data.get('category', 'uncategorized')
        qid = response_data.get('question_id')
        strat = response_data.get('strategy', 'unknown')
        
        prompts = [
            f"""
You are an evaluator. Task: {prompt.strip()}
Response: {response}
Rate 1–10. Return ONLY JSON {{"score":<1-10>,"reasoning":"..."}}
""",
            f"""
Evaluate quality on scale 1–10.
TEXT: {response[:300]}... (truncated)
Return only JSON with score and reasoning.
""",
            "Rate this content 1–10 for quality with 10 being the highest. Return ONLY JSON {\"score\":<1-10>,\"reasoning\":\"...\"}"  
        ]

        for idx, eval_prompt in enumerate(prompts):
            for attempt in range(self.retry_count):
                try:
                    completion = self.client.chat.completions.create(
                        model=self.deployment_name,
                        messages=[{"role": "user", "content": eval_prompt}],
                        temperature=0.3,
                        max_tokens=500
                    )
                    text = completion.choices[0].message.content.strip()
                    try:
                        result = json.loads(text)
                    except json.JSONDecodeError:
                        m = re.search(r"\{.*\}", text, re.DOTALL)
                        if m:
                            result = json.loads(m.group(0))
                        else:
                            raise ValueError("No JSON object")
                    if 'score' not in result:
                        raise ValueError("Missing 'score'")
                    reasoning = result.get('reasoning', '')
                    note = ''
                    if idx == 1:
                        note = ' (truncated)'
                    elif idx == 2:
                        note = ' (minimal)'
                    return {
                        'strategy': strat,
                        'question_id': qid,
                        'category': category,
                        'score': result['score'],
                        'reasoning': reasoning + note
                    }
                except Exception as e:
                    msg = str(e)
                    if 'content_filter' in msg and self.skip_filtered:
                        self.skipped_responses.append({'strategy': strat, 'question_id': qid, 'category': category, 'error': msg})
                        return None
                    if attempt < self.retry_count - 1:
                        time.sleep(self.retry_delay)
        # fallback neutral
        return {
            'strategy': strat,
            'question_id': qid,
            'category': category,
            'score': 5,
            'reasoning': 'Fallback neutral score after retries'
        }

    def evaluate_all(self, strategies: dict) -> list:
        results = []
        total = sum(len(resps) for resps in strategies.values())
        with tqdm(total=total, desc='Evaluating') as pbar:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {executor.submit(self.evaluate_response, resp): resp
                           for resps in strategies.values() for resp in resps}
                for fut in as_completed(futures):
                    out = fut.result()
                    if out is not None:
                        results.append(out)
                    pbar.update(1)
        if self.skipped_responses:
            print(f"Skipped {len(self.skipped_responses)} due to filtering.")
        return results

    def calculate_statistics(self, results: list) -> dict:
        stats = {'overall': {}, 'by_category': {}}
        strategies = set(r['strategy'] for r in results)
        for strat in strategies:
            scores = [r['score'] for r in results if r['strategy'] == strat]
            stats['overall'][strat] = {
                'average': float(np.mean(scores)) if scores else 0,
                'median': float(np.median(scores)) if scores else 0,
                'min': min(scores) if scores else 0,
                'max': max(scores) if scores else 0,
                'count': len(scores)
            }
            stats['by_category'][strat] = {}
            for cat in self.categories:
                cs = [r['score'] for r in results if r['strategy']==strat and r['category']==cat]
                stats['by_category'][strat][cat] = {'average': float(np.mean(cs)) if cs else 0, 'count': len(cs)}
        return stats

    def save_results(self, results: list, stats: dict, output_file: str):
        output = {'raw': results, 'stats': stats, 'skipped': self.skipped_responses}
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"Results saved to {output_file}")

    def print_summary(self, stats: dict):
        print("\n=== SUMMARY ===\n")
        print(f"{'Strategy':<20} {'Avg':<6} {'Med':<6} {'Min':<4} {'Max':<4} {'Count':<5}")
        print('-'*60)
        for strat, data in stats['overall'].items():
            print(f"{strat:<20} {data['average']:<6.2f} {data['median']:<6.2f} {data['min']:<4} {data['max']:<4} {data['count']:<5}")
        print("\nBY CATEGORY:\n")
        for strat, catstats in stats['by_category'].items():
            print(f"-- {strat} --")
            for cat, cd in catstats.items():
                print(f"  {cat:<15} avg={cd['average']:<5.2f} n={cd['count']}")


def score_with_azure(prompt: str, response: str, cfg: dict) -> float:
    """
    Convenience wrapper so benchmark.py can call azure_eval.score_with_azure(...)
    """
    az = cfg.get('azure', {})
    evaluator = LLMResponseEvaluator(
        api_key=az.get('api_key'),
        endpoint=az.get('endpoint'),
        api_version=az.get('api_version', '2023-05-15'),
        deployment_name=az.get('deployment_name'),
        max_workers=1,
        retry_count=1,
        retry_delay=1,
        skip_filtered=False
    )
    result = evaluator.evaluate_response({
        'prompt': prompt,
        'response': response,
        'category': None,
        'question_id': None,
        'strategy': None
    })
    return result.get('score') if result else None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate LLM responses using config-based Azure credentials')
    parser.add_argument('--config', default='config.json')
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', default='evaluation_results.json')
    parser.add_argument('--max-workers', type=int, default=5)
    parser.add_argument('--retry-count', type=int, default=3)
    parser.add_argument('--retry-delay', type=int, default=2)
    parser.add_argument('--skip-filtered', action='store_true')
    args = parser.parse_args()

    cfg = load_config(args.config)
    strategies = LLMResponseEvaluator(
        api_key=cfg['azure']['api_key'],
        endpoint=cfg['azure']['endpoint'],
        api_version=cfg['azure'].get('api_version', '2023-05-15'),
        deployment_name=cfg['azure']['deployment_name'],
        max_workers=args.max_workers,
        retry_count=args.retry_count,
        retry_delay=args.retry_delay,
        skip_filtered=args.skip_filtered
    ).load_data(args.input)

    evaluator = LLMResponseEvaluator(
        api_key=cfg['azure']['api_key'],
        endpoint=cfg['azure']['endpoint'],
        api_version=cfg['azure'].get('api_version', '2023-05-15'),
        deployment_name=cfg['azure']['deployment_name'],
        max_workers=args.max_workers,
        retry_count=args.retry_count,
        retry_delay=args.retry_delay,
        skip_filtered=args.skip_filtered
    )
    results = evaluator.evaluate_all(strategies)
    stats = evaluator.calculate_statistics(results)
    evaluator.save_results(results, stats, args.output)
    evaluator.print_summary(stats)
