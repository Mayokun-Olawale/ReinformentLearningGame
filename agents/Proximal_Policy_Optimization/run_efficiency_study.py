"""Reproducible PPO ablations, validation-only selection, seed repeats and speed tests.

Run from the project root with python -m agents.Proximal_Policy_Optimization.run_efficiency_study.
All training jobs run sequentially so their wall times are comparable.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MODULE = 'agents.Proximal_Policy_Optimization'
VARIANTS = [
    dict(id='model_1_baseline', name='PPO baseline', changes={}),
    dict(id='model_2_two_tick', name='Two-tick controls', changes=dict(action_repeat=2)),
    dict(id='model_3_curriculum', name='Curriculum', changes=dict(curriculum=True)),
    dict(id='model_4_lower_learning_rate', name='Lower learning rate', changes=dict(learning_rate=0.0001)),
    dict(id='model_5_longer_horizon', name='Longer discount horizon', changes=dict(gamma=0.995)),
    dict(id='model_6_smaller_network', name='Smaller network', changes=dict(hidden_size=64)),
]


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def best_validation(folder):
    rows = read(Path(folder)/'validation.json')
    return max(rows, key=lambda row: validation_score(row))


def validation_score(row):
    return (row['summary']['completion_rate'], row['summary']['mean_progress_pixels'])


def choose_combination(baseline, candidates):
    """No held-out reports enter the decision; strict improvement over baseline."""
    changes, accepted = {}, []
    threshold = validation_score(baseline)
    for spec, validation in candidates:
        if validation_score(validation) > threshold:
            changes.update(spec['changes'])
            accepted.append(spec['id'])
    return dict(id='model_7_combined', name='Validation-selected combination',
                changes=changes, accepted=accepted,
                selection_rule='Strictly higher best validation completion, then progress, than baseline')


def execute(module, args, log):
    env = dict(os.environ, SDL_VIDEODRIVER='dummy', SDL_AUDIODRIVER='dummy',
               MPLCONFIGDIR='/tmp/platformer-matplotlib', OMP_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    with Path(log).open('w') as stream:
        subprocess.run([sys.executable, '-u', '-m', module, *map(str, args)],
                       cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)


def config_for(base, changes, seed, budget):
    config = dict(base, **changes, seed=seed)
    config['total_timesteps'] = budget // config['action_repeat']
    config['validation_freq'] = 100000 // config['action_repeat']
    return config


def train_run(folder, config):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if (folder/'final.pt').exists() and (folder/'summary.json').exists():
        if read(folder/'config.json') != config:
            raise ValueError(f'Completed run has different config: {folder}')
        return
    if (folder/'metrics.csv').exists():
        raise ValueError(f'Incomplete run preserved at {folder}; move it aside before retrying')
    write(folder/'requested_config.json', config)
    print(f'Training {folder.name}: seed={config["seed"]}, repeat={config["action_repeat"]}', flush=True)
    execute(MODULE+'.train_ppo', ['--config', folder/'requested_config.json', '--run-dir', folder], folder/'training_console.log')
    execute(MODULE+'.plot_training', [folder], folder/'plot_console.log')


def evaluate_run(folder, seed_list, name='test', algo='ppo', checkpoint=None):
    folder = Path(folder)
    path = folder/(name+'.json')
    if path.exists():
        return read(path)
    print(f'Evaluating {folder.name}: {name}', flush=True)
    execute(MODULE+'.evaluate', ['--algo', algo, '--checkpoint', checkpoint or folder/'best.pt',
                                '--seed-list', seed_list, '--output', path], folder/(name+'_console.log'))
    result = read(path)
    print(f'{folder.name} {name}: {result["summary"]["completion_rate"]:.0%} completed', flush=True)
    return result


def snapshot(folder, baseline):
    paths = [*HERE.glob('*.py'), *HERE.glob('ppo/*.py'), *HERE.glob('config/*.json'),
             *ROOT.glob('Game/*.py'), ROOT/'game_env.py', ROOT/'requirements.txt',
             ROOT/'agents/Deep_Q-Network/training_env.py', ROOT/'agents/Deep_Q-Network/train_dqn.py']
    checksums = {}
    for source in paths:
        relative = source.relative_to(ROOT)
        target = folder/'source_snapshot'/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        checksums[str(relative)] = hashlib.sha256(source.read_bytes()).hexdigest()
    write(folder/'source_checksums.json', checksums)
    write(folder/'preserved_baseline_checksums.json', {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in baseline.iterdir() if p.is_file()})


def run_study(folder, baseline):
    folder, baseline = Path(folder).resolve(), Path(baseline).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    budget, seeds = 2000000, [7, 17, 27]
    base = read(HERE/'config/ppo.json')
    if not (folder/'plan.json').exists():
        plan = dict(nominal_physics_tick_budget=budget, training_seeds=seeds,
                    validation_seeds=list(range(100000, 100005)),
                    matched_dqn_test_seeds=list(range(400000, 400100)),
                    fresh_confirmation_seeds=list(range(900000, 900100)),
                    variants=VARIANTS, baseline_source=str(baseline),
                    selection='Best validation completion, then progress; no test-based tuning',
                    combination='Combine only individual variants strictly better than baseline on validation',
                    repeat_selection='Best baseline or variant including combined, ranked on seed-7 validation only',
                    speed_test='1, 4, 8 environments; fixed 512-transition rollout batches; three trials each',
                    timing='Sequential jobs; historical baseline wall time is contextual',
                    limitations=['Nominal ticks equal; actual ticks can differ at episode boundaries',
                                 'Action-repeat 2 uses twice the transitions and PPO updates',
                                 'Only five validation courses; selection can overfit',
                                 'DQN has one historical training seed; no cross-algorithm significance claim',
                                 'Fresh test seeds must not be used to choose or tune configurations'])
        write(folder/'plan.json', plan)
        write(folder/'test_seeds.json', plan['matched_dqn_test_seeds'])
        write(folder/'fresh_seeds.json', plan['fresh_confirmation_seeds'])
        snapshot(folder, baseline)
    expected = config_for(base, {}, 7, budget)
    if read(baseline/'config.json') != expected:
        raise ValueError('Historical baseline configuration differs from predeclared study baseline')
    baseline_dest = folder/'model_1_baseline'
    if not baseline_dest.exists():
        shutil.copytree(baseline, baseline_dest)
        write(baseline_dest/'origin.json', dict(source=str(baseline), preserved=True))
        # Historical 10-course report is retained as such, not mistaken for the study test.
        if (baseline_dest/'test.json').exists():
            (baseline_dest/'test.json').rename(baseline_dest/'historical_test.json')
    for spec in VARIANTS[1:]:
        train_run(folder/spec['id'], config_for(base, spec['changes'], 7, budget))
        write(folder/'status.json', dict(stage='ablations', completed=spec['id']))
    combined = choose_combination(best_validation(baseline_dest),
                                  [(spec, best_validation(folder/spec['id'])) for spec in VARIANTS[1:]])
    write(folder/'combination_selection.json', combined)
    all_specs = [*VARIANTS, combined]
    train_run(folder/combined['id'], config_for(base, combined['changes'], 7, budget))
    winner = max(all_specs, key=lambda spec: validation_score(best_validation(folder/spec['id'])))
    write(folder/'winner_selection.json', dict(winner=winner, selection_seed=7,
                                               validation=best_validation(folder/winner['id'])))
    print(f'Validation-selected configuration: {winner["id"]}', flush=True)
    for spec in (VARIANTS[0], winner) if winner['id'] != VARIANTS[0]['id'] else (VARIANTS[0],):
        for seed in seeds[1:]:
            train_run(folder/(spec['id']+f'_seed{seed}'), config_for(base, spec['changes'], seed, budget))
            write(folder/'status.json', dict(stage='seed_repeats', completed=spec['id'], seed=seed))
    # All configuration decisions are frozen before these held-out results are read.
    for spec in all_specs:
        evaluate_run(folder/spec['id'], folder/'test_seeds.json')
    for spec in (VARIANTS[0], winner) if winner['id'] != VARIANTS[0]['id'] else (VARIANTS[0],):
        for seed in seeds:
            dest = folder/(spec['id']+(f'_seed{seed}' if seed != 7 else ''))
            if seed != 7:
                evaluate_run(dest, folder/'test_seeds.json')
            evaluate_run(dest, folder/'fresh_seeds.json', 'fresh_test')
    dqn_root = ROOT/'agents/Deep_Q-Network/efficiency_study_2026_10_20'
    for model in ('model_2_gap_features', 'model_4_curriculum', 'model_7_combined'):
        checkpoint = dqn_root/model/'saved_models/dqn_platformer_best.zip'
        if checkpoint.exists():
            dest = folder/('dqn_'+model)
            dest.mkdir(exist_ok=True)
            evaluate_run(dest, folder/'test_seeds.json', algo='dqn', checkpoint=checkpoint)
            evaluate_run(dest, folder/'fresh_seeds.json', 'fresh_test', algo='dqn', checkpoint=checkpoint)
    execute(MODULE+'.benchmark', ['--checkpoint', folder/winner['id']/'best.pt',
                                 '--output', folder/'vectorization.json'], folder/'benchmark_console.log')
    execute(MODULE+'.summarize_efficiency_study', [folder], folder/'report_console.log')
    originals = read(folder/'preserved_baseline_checksums.json')
    for relative, expected_hash in originals.items():
        if hashlib.sha256((ROOT/relative).read_bytes()).hexdigest() != expected_hash:
            raise RuntimeError(f'Original baseline changed: {relative}')
    write(folder/'status.json', dict(stage='complete', winner=winner['id']))
    print(f'Study complete: {folder / "report.md"}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, default=HERE/'efficiency_study_2026_11_09')
    parser.add_argument('--baseline', type=Path, default=HERE/'runs/ppo_seed7')
    args = parser.parse_args()
    run_study(args.folder, args.baseline)
