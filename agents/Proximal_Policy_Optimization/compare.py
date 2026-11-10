"""Aggregate common-evaluator reports across independent training seeds."""
import argparse
import json
from pathlib import Path
import numpy as np


def compare(paths):
    reports = [json.loads(Path(path).read_text()) for path in paths]
    reference = reports[0]
    protocol = lambda r: (r['action_repeat'], r['max_episode_steps'], r['observation_version'],
                          r['evaluation_fruit_reward'], [e['seed'] for e in r['episodes']])
    if any(protocol(report) != protocol(reference) for report in reports):
        raise ValueError('Reports must share controls, horizon, rewards, observations and evaluation seeds')
    result = {}
    for algo in sorted({r['algo'] for r in reports}):
        group = [r for r in reports if r['algo'] == algo]
        metrics = {}
        for key in reference['summary']:
            values = [r['summary'][key] for r in group]
            metrics[key] = dict(mean=float(np.mean(values)),
                                std=float(np.std(values, ddof=1)) if len(values)>1 else None)
        result[algo] = dict(runs=len(group), model_decisions=[r['model_decisions'] for r in group],
                            metrics=metrics)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reports', nargs='+', type=Path)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent / 'runs' / 'comparison.json')
    args = parser.parse_args()
    result = compare(args.reports)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
