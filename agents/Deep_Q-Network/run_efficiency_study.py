"""Train the predeclared variants and archive matched-course evidence separately."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from training_paths import PROJECT_ROOT, TRAINING_DIR

VARIANTS = [
    dict(id='model_1_baseline', name='Original DQN', method='Original 51 inputs', source='saved_models', legacy=True, repeat=4, rate=0.0003, curriculum=False, fruit=2),
    dict(id='model_2_gap_features', name='Gap features', method='Exact gap and landing observations (60 inputs)', source='model_2_gap_features/saved_models', repeat=4, rate=0.0003, curriculum=False, fruit=2),
    dict(id='model_3_two_tick', name='Two-tick decisions', method='Gap features + decisions every two physics ticks', repeat=2, rate=0.0003, curriculum=False, fruit=2),
    dict(id='model_4_curriculum', name='Curriculum', method='Gap features + gradually harder training courses', repeat=4, rate=0.0003, curriculum=True, fruit=2),
    dict(id='model_5_lower_learning_rate', name='Lower learning rate', method='Gap features + learning rate 0.0001 instead of 0.0003', repeat=4, rate=0.0001, curriculum=False, fruit=2),
    dict(id='model_6_fruit_reward', name='Larger fruit bonus', method='Gap features + training fruit bonus 4 instead of 2', repeat=4, rate=0.0003, curriculum=False, fruit=4),
    dict(id='model_7_combined', name='Combined methods', method='Gap features + two-tick decisions + curriculum + lower learning rate + fruit bonus 4', repeat=2, rate=0.0001, curriculum=True, fruit=4),
]


def run_study(folder, workers=2):
    folder = Path(folder).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'plan.json').exists():
        raise ValueError('This study already has a plan. Choose a new folder to avoid overwriting evidence.')
    if not (folder / 'preserved_originals.json').exists():
        originals = []
        for name in ('saved_models', 'tensorboard_logs', 'walkthrough', 'model_2_gap_features', 'gap_experiment'):
            originals.extend(p for p in (TRAINING_DIR/name).rglob('*') if p.is_file())
        originals.extend(TRAINING_DIR/name for name in ('evaluation_results.json', 'training_results.png'))
        checksums = {str(p.relative_to(PROJECT_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in originals if p.is_file()}
        (folder/'preserved_originals.json').write_text(json.dumps(checksums, indent=2)+'\n')
    plan = dict(seed=7, nominal_physics_tick_budget=2000000,
                validation_seeds=[100000, 100004], validation_every_nominal_ticks=100000,
                test_seeds=[400000, 400099], episodes=100,
                evaluation_fruit_reward=2, checkpoint_selection='validation completions, then mean progress',
                variants=VARIANTS,
                limitations=['One training seed per method',
                             'Two-tick runs use twice as many decisions and gradient updates for a comparable nominal physics budget',
                             'The original baseline was resumed during its historical training run',
                             'Rewards in the fruit-bonus training logs differ; test rewards always use the original formula'])
    (folder / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    snapshot = folder / 'source_snapshot'
    shutil.copytree(PROJECT_ROOT / 'Game', snapshot / 'Game', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copy2(PROJECT_ROOT / 'game_env.py', snapshot / 'game_env.py')
    shutil.copy2(PROJECT_ROOT / 'requirements.txt', snapshot / 'requirements.txt')
    scripts = snapshot / 'agents/Deep_Q-Network'
    scripts.mkdir(parents=True)
    for source in [*TRAINING_DIR.glob('*.py'), TRAINING_DIR/'requirements-training.txt']:
        shutil.copy2(source, scripts/source.name)
    env = dict(os.environ, SDL_VIDEODRIVER='dummy', MPLCONFIGDIR='/tmp/platformer-matplotlib',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')

    def execute(args, log):
        with log.open('w') as stream:
            subprocess.run([sys.executable, '-u', *map(str, args)], cwd=PROJECT_ROOT,
                           env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)

    def experiment(spec):
        dest = folder / spec['id']
        dest.mkdir(exist_ok=False)
        (dest / 'method.json').write_text(json.dumps(spec, indent=2) + '\n')
        if 'source' in spec:
            shutil.copytree(TRAINING_DIR / spec['source'], dest / 'saved_models')
            origin = TRAINING_DIR if spec.get('legacy') else TRAINING_DIR / 'model_2_gap_features'
            shutil.copytree(origin / 'tensorboard_logs', dest / 'tensorboard_logs')
            shutil.copy2(origin / 'training_results.png', dest / 'training_results.png')
            shutil.copy2(origin / 'evaluation_results.json', dest / 'historical_evaluation_results.json')
        else:
            print(f"Training {spec['id']}", flush=True)
            args = [TRAINING_DIR/'train_dqn.py', '--timesteps', 2000000//spec['repeat'], '--seed', 7,
                    '--action-repeat', spec['repeat'], '--learning-rate', spec['rate'], '--fruit-reward', spec['fruit'],
                    '--validation-freq', 100000//spec['repeat'], '--model-dir', dest/'saved_models',
                    '--log-dir', dest/'tensorboard_logs', '--chart-path', dest/'training_results.png']
            if spec['curriculum']:
                args.append('--curriculum')
            execute(args, dest/'training_console.log')
        model = dest/'saved_models/dqn_platformer_best.zip'
        print(f"Evaluating {spec['id']}", flush=True)
        args = [TRAINING_DIR/'evaluate_dqn.py', '--model', model, '--episodes', 100,
                '--seed', 400000, '--output', dest/'evaluation_results.json']
        if spec.get('legacy'):
            args.append('--legacy-observations')
        execute(args, dest/'evaluation_console.log')
        evaluation = json.loads((dest/'evaluation_results.json').read_text())
        successes = [r for r in evaluation['episodes'] if r['outcome']=='completed']
        example = successes[0] if successes else max(evaluation['episodes'], key=lambda r: (r['progress_pixels'], -r['ticks']))
        args = [TRAINING_DIR/'record_walkthrough.py', '--model', model, '--folder', dest/'walkthrough',
                '--seed', example['seed'], '--speed', 3]
        if spec.get('legacy'):
            args.append('--legacy-observations')
        execute(args, dest/'recording_console.log')
        recording = json.loads((dest/'walkthrough/walkthrough.json').read_text())
        for key in ('outcome', 'ticks', 'fruit', 'damage', 'progress_pixels'):
            if recording[key] != example[key]:
                raise ValueError(f"Recording disagrees with evaluation: {spec['id']} {key}")
        print(f"Finished {spec['id']}: {json.dumps(evaluation['summary'])}", flush=True)
        return spec['id']

    # Start the long new runs first. Threads coordinate separate training processes.
    ordered = [s for s in VARIANTS if 'source' not in s] + [s for s in VARIANTS if 'source' in s]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = {pool.submit(experiment, spec): spec['id'] for spec in ordered}
        for job in as_completed(jobs):
            job.result()
    print('All seven model evidence folders are ready.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, required=True)
    parser.add_argument('--workers', type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    run_study(args.folder, args.workers)
