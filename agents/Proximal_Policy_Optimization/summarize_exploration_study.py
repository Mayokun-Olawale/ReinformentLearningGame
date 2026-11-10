"""Exploration study results with validation-only decisions and seed uncertainty."""
import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from agents.Proximal_Policy_Optimization.run_efficiency_study import read, write, best_validation
from agents.Proximal_Policy_Optimization.run_exploration_study import BASELINE, VARIANTS


def summarize(folder):
    folder=Path(folder)
    selection=read(folder/'selection.json')
    rows=[]
    for dest in sorted(folder.iterdir()):
        if not dest.is_dir() or not (dest/'test.json').exists(): continue
        test=read(dest/'test.json')
        config=read(dest/'config.json') if (dest/'config.json').exists() else {}
        val=best_validation(dest) if (dest/'validation.json').exists() else None
        row=dict(model=dest.name,seed=config.get('seed'),
                 entropy_start=config.get('entropy_coef'),
                 entropy_end=config.get('entropy_final_coef',config.get('entropy_coef')),
                 entropy_schedule=config.get('entropy_schedule','constant'),
                 validation_completion=val['summary']['completion_rate'] if val else None,
                 selected_decisions=test['model_decisions'],
                 test_completion=test['summary']['completion_rate'],
                 mean_fruit=test['summary']['mean_fruit'],mean_damage=test['summary']['mean_damage'],
                 falls=test['summary']['falls'],timeouts=test['summary']['timeouts'],
                 defeated=sum(e['outcome']=='defeated' for e in test['episodes']))
        if (dest/'metrics.csv').exists():
            with (dest/'metrics.csv').open() as stream: metrics=list(csv.DictReader(stream))
            row['initial_policy_entropy']=float(np.mean([float(r['entropy']) for r in metrics[:10]]))
            row['final_policy_entropy']=float(np.mean([float(r['entropy']) for r in metrics[-10:]]))
        rows.append(row)
    write(folder/'results.json',rows)
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (folder/'results.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    names=list(dict.fromkeys([BASELINE,selection['winner']]))
    aggregates={}
    for name in names:
        members=[r for r in rows if r['model']==name or r['model'].startswith(name+'_seed')]
        values=[r['test_completion'] for r in members]
        aggregates[name]=dict(training_seeds=[r['seed'] for r in members],
            mean_completion=float(np.mean(values)),
            std_completion=float(np.std(values,ddof=1)) if len(values)>1 else None,
            per_seed_completion={str(r['seed']):r['test_completion'] for r in members})
    write(folder/'seed_aggregates.json',aggregates)
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for name in [BASELINE,*[s['id'] for s in VARIANTS]]:
        dest=folder/name
        config=read(dest/'config.json');history=read(dest/'validation.json')
        with (dest/'metrics.csv').open() as stream: metrics=list(csv.DictReader(stream))
        label=name.replace('entropy_','').replace('_',' ')
        axes[0,0].plot([r['steps'] for r in history],
                      [r['summary']['completion_rate'] for r in history],label=label,marker='.')
        steps=[int(r['steps']) for r in metrics]
        axes[0,1].plot(steps,[float(r['entropy']) for r in metrics],label=label)
        axes[1,0].plot(steps,[float(r.get('entropy_coef',config['entropy_coef'])) for r in metrics],label=label)
        # Completion always stays distinct from reward/fruit shaping.
        axes[1,1].plot([r['steps'] for r in history],
                      [r['summary']['mean_fruit'] for r in history],label=label)
    for axis,ylabel in zip(axes.flatten(),['25-course validation completion','Policy entropy',
                                          'Entropy bonus coefficient','Validation mean fruit']):
        axis.set(xlabel='Environment decisions',ylabel=ylabel);axis.grid(alpha=.2)
    axes[0,0].legend();fig.tight_layout();fig.savefig(folder/'exploration_curves.png',dpi=160);plt.close(fig)
    lines=['# PPO exploration tuning','',
        'Only the entropy bonus varies. All new runs start from scratch with 500,000 decisions '
        '(2 million nominal physics ticks), four environments and the original controls, rewards, '
        'learning rate, batch size, network and discount factor. Curriculum stays off.','',
        'Checkpoint selection uses 25 previously unused validation courses (110000–110024), '
        'every approximately 50,000 decisions. Existing baseline checkpoints are reselected on '
        'the identical validation set and cadence. The final 100 courses (1000000–1000099) '
        'are opened only after configuration decisions are frozen.','',
        '| Run | Entropy coefficient | Validation completion | Final-test completion | Fruit | Falls | Defeated | Timeouts |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        entropy='DQN reference' if row['entropy_start'] is None else (
            f"{row['entropy_start']} → {row['entropy_end']} (20% hold)" if row['entropy_schedule']=='linear'
            else str(row['entropy_start']))
        val=f"{row['validation_completion']:.0%}" if row['validation_completion'] is not None else '—'
        lines.append(f"| {row['model']} | {entropy} | {val} | {row['test_completion']:.0%} | {row['mean_fruit']:.2f} | {row['falls']} | {row['defeated']} | {row['timeouts']} |")
    lines+=['','## Selection and confirmation','',
            f"Validation-selected configuration: **{selection['winner']}**. "
            f"New candidate beat baseline validation: **{selection['improved_on_validation']}**. "
            'Final-test results were not used to select or modify the configuration.','']
    for name,stats in aggregates.items():
        spread=f" ± {stats['std_completion']:.1%}" if stats['std_completion'] is not None else ''
        lines.append(f"- {name}: {stats['mean_completion']:.1%}{spread} completion across training seeds {stats['training_seeds']}; per-seed results {stats['per_seed_completion']}.")
    lines+=['','## Interpretation limits','',
        '- Candidate screening uses one pilot training seed. Only a new validation winner receives two more training runs; baseline weights are independently trained seeds 7, 17 and 27.',
        '- Standard deviations describe variation across training seeds; they are not confidence intervals.',
        '- The baseline uses existing trajectories. Its original training time used five-course validation; new runs use 25, so their elapsed training times are not directly comparable.',
        '- Historical PPO figures (15% on 400000-series courses, 40% on ten 200000-series courses) are different checkpoint-selection/test protocols. Use the baseline in this report for exploration comparisons.',
        '- The DQN curriculum reference is an existing single-seed checkpoint, evaluated on the same fresh final courses.',
        '- High entropy or fruit collection alone does not establish successful navigation. Completion is the primary metric.',
        '- The 100 final courses must not be reused to tune the next study. Original checkpoints and frozen source files are preserved.', '',
        'Artifacts: `plan.json`, `selection.json`, `results.csv`, `seed_aggregates.json`, '
        '`exploration_curves.png`, per-run checkpoints, validation/test episodes and entropy/action diagnostics.','']
    (folder/'report.md').write_text('\n'.join(lines))
    print('\n'.join(lines))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder',type=Path)
    summarize(parser.parse_args().folder)
