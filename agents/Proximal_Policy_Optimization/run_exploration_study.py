"""Tune only PPO entropy, with larger validation and untouched final test seeds."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

from agents.Proximal_Policy_Optimization.run_efficiency_study import (
    HERE, ROOT, MODULE, read, write, execute, train_run, evaluate_run,
    best_validation, validation_score, snapshot)

VARIANTS = [
    dict(id='entropy_zero', changes=dict(entropy_coef=0.0)),
    dict(id='entropy_stronger', changes=dict(entropy_coef=0.03)),
    dict(id='entropy_hold_decay', changes=dict(entropy_coef=0.05, entropy_schedule='linear',
         entropy_hold_fraction=0.2, entropy_final_coef=0.001)),
]
BASELINE = 'entropy_baseline'


def candidate_config(changes, seed):
    return dict(read(HERE/'config/ppo.json'), **changes, seed=seed,
                validation_seed_start=110000, validation_episodes=25, validation_freq=50000)


def checkpoint_subset(source):
    """Match new training's 50k validation cadence, including the final rollout."""
    paths = sorted(Path(source).glob('checkpoint_*.pt'), key=lambda p: int(p.stem.split('_')[-1]))
    next_step = 50000
    chosen = []
    for path in paths:
        step = int(path.stem.split('_')[-1])
        if step >= next_step or step == 500000:
            chosen.append(path)
            next_step = step + 50000
    if not chosen or int(chosen[-1].stem.split('_')[-1]) != 500000:
        raise ValueError(f'Missing completed historical checkpoint sequence: {source}')
    return chosen


def reselect(source, folder):
    """No learning on validation: evaluate and choose existing immutable weights."""
    import torch
    from agents.Proximal_Policy_Optimization.evaluate import evaluate
    torch.set_num_threads(1)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    config = candidate_config({}, read(Path(source)/'config.json')['seed'])
    write(folder/'config.json', config)
    shutil.copy2(Path(source)/'config.json', folder/'original_training_config.json')
    history, best = [], (-1, -1)
    started = time.perf_counter()
    for path in checkpoint_subset(source):
        report = evaluate(path, seeds=list(range(110000,110025)))
        report['steps'] = report['model_decisions']
        report['elapsed_seconds'] = time.perf_counter()-started
        history.append(report)
        if validation_score(report) > best:
            best = validation_score(report)
            shutil.copy2(path, folder/'best.pt')
            selected = path
        write(folder/'validation.json', history)
        print(f'Reselected {path.name}: {report["summary"]["completion_rate"]:.0%}', flush=True)
    shutil.copy2(Path(source)/'summary.json', folder/'summary.json')
    shutil.copy2(Path(source)/'metrics.csv', folder/'metrics.csv')
    write(folder/'reselection.json', dict(source=str(source), selected_checkpoint=str(selected),
        selection_seconds=time.perf_counter()-started,
        selection_seeds=list(range(110000,110025)),
        checkpoint_candidates=[str(p) for p in checkpoint_subset(source)],
        note='Existing training trajectories reused; only checkpoint selection changed. Training time excludes reselection.'))


def reselect_run(source, dest):
    if (dest/'reselection.json').exists():
        return
    print(f'Reselecting existing baseline: {dest.name}', flush=True)
    dest.mkdir(parents=True, exist_ok=True)
    execute(MODULE+'.run_exploration_study', ['--reselect-source', source, '--folder', dest], dest/'reselection_console.log')


def run(folder):
    folder = Path(folder).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    baseline = HERE/'runs/ppo_seed7'
    old = HERE/'efficiency_study_2026_11_09'
    sources = {7:baseline,17:old/'model_1_baseline_seed17',27:old/'model_1_baseline_seed27'}
    if not (folder/'plan.json').exists():
        write(folder/'plan.json', dict(training_seeds=[7,17,27],
            total_decisions=500000,nominal_physics_ticks=2000000,action_repeat=4,
            pilot_seed=7,variants=VARIANTS, baseline_entropy_coef=.01,
            validation_seeds=list(range(110000,110025)),validation_freq=50000,
            final_test_seeds=list(range(1000000,1000100)),
            selection='Best 25-course validation completion, then progress; no final-test tuning',
            confirmation='Repeat best new entropy setting with seeds 17 and 27 only if it strictly beats baseline validation; reselect baseline for all three seeds',
            baseline_reuse='Identical existing training trajectories, with checkpoints reselected on new validation set at matching 50k cadence',
            held_fixed=['observations','controls','reward','course distribution','learning rate',
                        'gamma','batch size','rollout length','epochs','network','nominal physics budget'],
            limitations=['Single pilot seed for candidate selection','Three seeds offer limited uncertainty estimates',
                         'Validation changed from historical study; historical completion figures are not the new baseline',
                         'Fresh final tests must not be reused for tuning']))
        write(folder/'test_seeds.json', list(range(1000000,1000100)))
        snapshot(folder, baseline)
        write(folder/'original_checkpoint_hashes.json', {
            str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for source in sources.values() for p in checkpoint_subset(source)})
    reselect_run(sources[7],folder/BASELINE)
    for spec in VARIANTS:
        train_run(folder/spec['id'], candidate_config(spec['changes'],7))
        write(folder/'status.json',dict(stage='pilot',completed=spec['id']))
    baseline_score = validation_score(best_validation(folder/BASELINE))
    best_spec = max(VARIANTS,key=lambda s:validation_score(best_validation(folder/s['id'])))
    improved = validation_score(best_validation(folder/best_spec['id'])) > baseline_score
    winner = best_spec['id'] if improved else BASELINE
    write(folder/'selection.json',dict(winner=winner,improved_on_validation=improved,
        best_new_candidate=best_spec,baseline_validation=best_validation(folder/BASELINE),
        candidate_validation=best_validation(folder/best_spec['id']),
        decision_before_final_tests=True))
    print(f'Validation-selected exploration: {winner}',flush=True)
    for seed in [17,27]:
        reselect_run(sources[seed],folder/(BASELINE+f'_seed{seed}'))
        if improved:
            train_run(folder/(winner+f'_seed{seed}'),candidate_config(best_spec['changes'],seed))
        write(folder/'status.json',dict(stage='confirmation',completed_seed=seed))
    # All choices are frozen before final tests are opened.
    names = [BASELINE,*[s['id'] for s in VARIANTS],BASELINE+'_seed17',BASELINE+'_seed27']
    if improved:
        names += [winner+'_seed17',winner+'_seed27']
    write(folder/'status.json',dict(stage='final_evaluation',winner=winner))
    for name in names:
        evaluate_run(folder/name,folder/'test_seeds.json')
    dqn = ROOT/'agents/Deep_Q-Network/efficiency_study_2026_10_20/model_4_curriculum/saved_models/dqn_platformer_best.zip'
    if dqn.exists():
        dest = folder/'dqn_curriculum_reference'
        dest.mkdir(exist_ok=True)
        evaluate_run(dest,folder/'test_seeds.json',algo='dqn',checkpoint=dqn)
    execute(MODULE+'.summarize_exploration_study',[folder],folder/'report_console.log')
    for relative, digest in read(folder/'original_checkpoint_hashes.json').items():
        if hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()!=digest:
            raise RuntimeError('Original checkpoint changed: '+relative)
    write(folder/'status.json',dict(stage='complete',winner=winner))
    print('Exploration study complete: '+str(folder/'report.md'),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder',type=Path,default=HERE/'exploration_study_2026_11_09')
    parser.add_argument('--reselect-source',type=Path)
    args=parser.parse_args()
    if args.reselect_source:
        reselect(args.reselect_source,args.folder)
    else:
        run(args.folder)
