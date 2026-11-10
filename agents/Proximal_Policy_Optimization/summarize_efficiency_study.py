"""Produce auditable tables, seed aggregates and sample/time learning curves."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from agents.Proximal_Policy_Optimization.run_efficiency_study import read, write, best_validation


def summarize(folder):
    folder = Path(folder)
    selection = read(folder/'winner_selection.json')
    winner = selection['winner']['id']
    rows = []
    for dest in sorted(folder.iterdir()):
        if not dest.is_dir() or not (dest/'test.json').exists():
            continue
        test = read(dest/'test.json')
        config = read(dest/'config.json') if (dest/'config.json').exists() else {}
        summary = read(dest/'summary.json') if (dest/'summary.json').exists() else {}
        validation = best_validation(dest) if (dest/'validation.json').exists() else None
        row = dict(model=dest.name, seed=config.get('seed', 7),
                   action_repeat=test['action_repeat'], best_checkpoint_decisions=test['model_decisions'],
                   validation_completion=validation['summary']['completion_rate'] if validation else None,
                   completion=test['summary']['completion_rate'], fruit=test['summary']['mean_fruit'],
                   damage=test['summary']['mean_damage'], falls=test['summary']['falls'],
                   timeouts=test['summary']['timeouts'],
                   training_seconds=summary.get('elapsed_seconds'),
                   total_decisions=summary.get('model_decisions'),
                   actual_physics_ticks=summary.get('actual_physics_ticks'))
        if (dest/'fresh_test.json').exists():
            row['fresh_completion'] = read(dest/'fresh_test.json')['summary']['completion_rate']
        rows.append(row)
    write(folder/'results.json', rows)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (folder/'results.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    aggregates = {}
    for name in dict.fromkeys(['model_1_baseline', winner]):
        members = [r for r in rows if r['model']==name or r['model'].startswith(name+'_seed')]
        aggregates[name] = dict(training_seeds=[r['seed'] for r in members])
        for key in ('completion', 'fresh_completion', 'fruit', 'damage', 'training_seconds'):
            values = [r[key] for r in members if r.get(key) is not None]
            aggregates[name][key] = dict(mean=float(np.mean(values)),
                std=float(np.std(values, ddof=1)) if len(values)>1 else None)
    write(folder/'seed_aggregates.json', aggregates)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for dest in sorted(folder.glob('model_*')):
        if '_seed' in dest.name or not (dest/'validation.json').exists():
            continue
        history, config = read(dest/'validation.json'), read(dest/'config.json')
        label = dest.name.replace('model_', '').replace('_', ' ')
        ticks = [r['steps']*config['action_repeat'] for r in history]
        seconds = [r['elapsed_seconds'] for r in history]
        completion = [r['summary']['completion_rate'] for r in history]
        fruit = [r['summary']['mean_fruit'] for r in history]
        axes[0, 0].plot(ticks, completion, label=label)
        axes[0, 1].plot(seconds, completion, label=label)
        axes[1, 0].plot(ticks, fruit, label=label)
        axes[1, 1].plot(seconds, fruit, label=label)
    for axis, xlabel, ylabel in [(axes[0,0], 'Nominal physics ticks', 'Validation completion'),
                                 (axes[0,1], 'Elapsed seconds', 'Validation completion'),
                                 (axes[1,0], 'Nominal physics ticks', 'Validation mean fruit'),
                                 (axes[1,1], 'Elapsed seconds', 'Validation mean fruit')]:
        axis.set(xlabel=xlabel, ylabel=ylabel)
        axis.grid(alpha=.2)
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(folder/'learning_curves.png', dpi=160)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(12, 5))
    axis.bar([r['model'].replace('model_', '').replace('_',' ') for r in rows], [r['completion']*100 for r in rows])
    axis.set(ylabel='Completion on 100 matched DQN courses (%)', ylim=(0,100))
    axis.tick_params(axis='x', rotation=65)
    fig.tight_layout()
    fig.savefig(folder/'matched_completion.png', dpi=160)
    plt.close(fig)
    lines = ['# PPO efficiency study', '',
             'Each PPO run has a budget of 2,000,000 nominal physics ticks. '
             'Two-tick controls receive 1,000,000 decisions; four-tick controls receive 500,000. '
             'Actual ticks are recorded separately because episodes can end partway through a decision.', '',
             f'Configuration selected using seed-7 validation only: **{winner}**. '
             'Checkpoint selection uses completion then progress on five fixed full-difficulty courses. '
             'Held-out test results do not enter configuration selection.', '',
             '| Run | Validation completion | Matched test completion | Fruit | Falls | Timeouts | Training seconds |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        val = f"{r['validation_completion']:.0%}" if r['validation_completion'] is not None else '—'
        seconds = f"{r['training_seconds']:.1f}" if r['training_seconds'] is not None else 'historical DQN'
        lines.append(f"| {r['model']} | {val} | {r['completion']:.0%} | {r['fruit']:.2f} | {r['falls']} | {r['timeouts']} | {seconds} |")
    lines += ['', '## Three-seed confirmation', '',
              'Values below are mean ± sample standard deviation across independently trained seeds 7, 17 and 27. '
              'Matched courses use seeds 400000–400099; fresh confirmation uses 900000–900099.', '']
    for name, stats in aggregates.items():
        values = []
        for key in ('completion', 'fresh_completion'):
            metric = stats[key]
            spread = f" ± {metric['std']:.1%}" if metric['std'] is not None else ''
            values.append(f"{key}: {metric['mean']:.1%}{spread}")
        lines.append(f"- {name}: {'; '.join(values)}.")
    if (folder/'vectorization.json').exists():
        bench = read(folder/'vectorization.json')
        lines += ['', '## CPU throughput', '',
                  'Fixed 512-transition rollout batches and 8192 measured decisions per trial, '
                  'after 2048 warmup decisions; three trials with rotated execution order. '
                  'These exclude validation and checkpoint I/O and do not establish equal learned skill.', '']
        for count, stats in bench['summary'].items():
            lines.append(f"- {count} environments: {stats['mean_decisions_per_second']:.0f} ± {stats['std_decisions_per_second']:.0f} decisions/second.")
    lines += ['', '## Limits', '',
              '- Historical baseline training time was measured before this study; new runs execute sequentially.',
              '- Five validation courses can produce noisy selections. The combined method is tested rather than assumed to improve.',
              '- The DQN checkpoints are existing 60-input gap-feature, curriculum (the original DQN study winner), and combined models, re-evaluated using the current engine. They each have only one training seed.',
              '- Action repeat is preserved per model. All methods see the same layouts, features, reward formula and 18000-tick horizon; control frequencies can differ.',
              '- Equal nominal physics budgets do not imply equal transitions, PPO updates or wall time. Learning curves show elapsed time separately.',
              '- Three seeds provide limited uncertainty estimates; the reported spread is across training runs, not a confidence interval.',
              '- Test seeds 400000–400099 were used in the earlier DQN study. Fresh confirmation is reported separately and must not be used for tuning.', '',
              'Artifacts: `results.csv`, `results.json`, `seed_aggregates.json`, `learning_curves.png`, '
              '`matched_completion.png`, `vectorization.json`, per-run checkpoints/logs, '
              '`source_snapshot`, and checksums preserving the original PPO run.', '']
    (folder/'report.md').write_text('\n'.join(lines))
    print('\n'.join(lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    summarize(parser.parse_args().folder)
