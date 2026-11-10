"""Train custom PyTorch PPO on independent headless platformer instances."""
import argparse
import csv
import json
import math
import time
from pathlib import Path
import torch
from agents.Proximal_Policy_Optimization.envs import make_env
from agents.Proximal_Policy_Optimization.ppo.trainer import PPOTrainer
from agents.Proximal_Policy_Optimization.evaluate import evaluate_policy

DEFAULT_RUN_DIR = Path(__file__).resolve().parent / 'runs' / 'ppo'
DEFAULT_CONFIG = Path(__file__).parent / 'config' / 'ppo.json'


def entropy_coefficient(config, progress):
    """Optional hold then linear decay; legacy configs retain constant entropy."""
    start = config['entropy_coef']
    if config.get('entropy_schedule', 'constant') == 'constant':
        return start
    hold = config.get('entropy_hold_fraction', 0.0)
    fraction = min(1.0, max(0.0, (progress-hold)/(1-hold)))
    return start + fraction * (config['entropy_final_coef']-start)


def validate(config):
    for key in ('total_timesteps', 'num_envs', 'rollout_steps', 'epochs', 'minibatch_size',
                'hidden_size', 'max_episode_steps', 'validation_freq', 'validation_episodes'):
        if not isinstance(config[key], int) or isinstance(config[key], bool) or config[key] <= 0:
            raise ValueError(f'{key} must be a positive integer')
    if config['total_timesteps'] % config['num_envs']:
        raise ValueError('total_timesteps must be divisible by num_envs for an exact transition budget')
    for key in ('learning_rate', 'clip', 'max_grad_norm'):
        if not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f'{key} must be finite and positive')
    for key in ('gamma', 'gae_lambda'):
        if not 0 <= config[key] <= 1:
            raise ValueError(f'{key} must be in [0, 1]')
    for key in ('value_coef', 'entropy_coef', 'fruit_reward', 'target_kl'):
        if not math.isfinite(config[key]) or config[key] < 0:
            raise ValueError(f'{key} must be finite and nonnegative')
    if config['value_clip'] is not None and (not math.isfinite(config['value_clip']) or config['value_clip'] <= 0):
        raise ValueError('value_clip must be null or positive and finite')
    if config['action_repeat'] not in (1, 2, 4):
        raise ValueError('action_repeat must be 1, 2 or 4')
    if config.get('entropy_schedule', 'constant') not in ('constant', 'linear'):
        raise ValueError('entropy_schedule must be constant or linear')
    if not 0 <= config.get('entropy_hold_fraction', 0.0) < 1:
        raise ValueError('entropy_hold_fraction must be in [0, 1)')
    if config.get('entropy_schedule') == 'linear':
        final = config.get('entropy_final_coef', -1)
        if not math.isfinite(final) or final < 0:
            raise ValueError('entropy_final_coef must be finite and nonnegative')
    validation_start = config.get('validation_seed_start', 100000)
    if not isinstance(validation_start, int) or validation_start < 100000:
        raise ValueError('validation_seed_start must be an integer outside training seeds')


def train(config=None, run_dir=DEFAULT_RUN_DIR, resume=None):
    settings = json.loads(DEFAULT_CONFIG.read_text())
    settings.update(config or {})
    validate(settings)
    torch.set_num_threads(1)
    folder = Path(run_dir)
    folder.mkdir(parents=True, exist_ok=True)
    if (folder/'metrics.csv').exists():
        raise ValueError('Use a fresh run directory to preserve existing results')
    (folder/'config.json').write_text(json.dumps(settings, indent=2)+'\n')
    envs = []
    try:
        for i in range(settings['num_envs']):
            envs.append(make_env(settings, i))
        trainer = PPOTrainer(envs, settings)
        if resume:
            restored = PPOTrainer.load(resume, envs, settings['device'])
            # Resume optimizer/policy; simulator episodes restart, as documented.
            for key in settings:
                if key not in ('total_timesteps', 'device', 'validation_freq', 'validation_episodes', 'validation_seed_start') and settings[key] != restored.config.get(key):
                    raise ValueError(f'Resume configuration mismatch: {key}')
            trainer = restored
            trainer.config = settings
        target = trainer.steps + settings['total_timesteps']
        next_validation = trainer.steps + settings['validation_freq']
        best = (-1, -1)
        history = []
        started = time.perf_counter()
        initial_steps = trainer.steps
        with (folder/'metrics.csv').open('w', newline='') as stream:
            writer = None
            while trainer.steps < target:
                progress = (trainer.steps-initial_steps)/settings['total_timesteps']
                entropy_coef = entropy_coefficient(settings, progress)
                if settings['linear_lr_decay']:
                    fraction = 1-(trainer.steps-initial_steps)/settings['total_timesteps']
                    for group in trainer.optimizer.param_groups:
                        group['lr'] = settings['learning_rate'] * fraction
                tick = time.perf_counter()
                rollout = trainer.collect_rollout(min(settings['rollout_steps'],
                                                      (target-trainer.steps)//settings['num_envs']))
                collection_seconds = time.perf_counter()-tick
                tick = time.perf_counter()
                stats = trainer.update(rollout, entropy_coef=entropy_coef)
                row = dict(steps=trainer.steps, physics_ticks=trainer.physics_ticks,
                           elapsed_seconds=time.perf_counter()-started,
                           steps_per_second=(trainer.steps-initial_steps)/(time.perf_counter()-started),
                           collection_seconds=collection_seconds, update_seconds=time.perf_counter()-tick,
                           learning_rate=trainer.optimizer.param_groups[0]['lr'],
                           entropy_coef=entropy_coef, **stats,
                           **{f'action_fraction_{i}': float((rollout.actions == i).float().mean())
                              for i in range(envs[0].action_space.n)})
                if writer is None:
                    writer = csv.DictWriter(stream, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                stream.flush()
                trainer.save_checkpoint(folder/'latest.pt')
                print(f"PPO {trainer.steps}/{target} decisions; {row['steps_per_second']:.0f} decisions/s", flush=True)
                if trainer.steps >= next_validation or trainer.steps == target:
                    def policy(obs):
                        with torch.no_grad():
                            return trainer.model(trainer.tensor(obs).unsqueeze(0))[0].argmax(-1).item()
                    validation_start = settings.get('validation_seed_start', 100000)
                    report = evaluate_policy(policy, settings, list(range(validation_start, validation_start+settings['validation_episodes'])))
                    report.update(steps=trainer.steps, elapsed_seconds=time.perf_counter()-started)
                    history.append(report)
                    (folder/'validation.json').write_text(json.dumps(history, indent=2)+'\n')
                    summary = report['summary']
                    score = (summary['completion_rate'], summary['mean_progress_pixels'])
                    if score > best:
                        best = score
                        trainer.save_checkpoint(folder/'best.pt')
                    trainer.save_checkpoint(folder/f'checkpoint_{trainer.steps}.pt')
                    next_validation = trainer.steps + settings['validation_freq']
                    print(f"Validation: {summary['completion_rate']:.0%} completed; mean progress {summary['mean_progress_pixels']:.0f}px", flush=True)
                (folder/'episodes.json').write_text(json.dumps(trainer.episodes, indent=2)+'\n')
        trainer.save_checkpoint(folder/'final.pt')
        (folder/'summary.json').write_text(json.dumps(dict(
            model_decisions=trainer.steps, actual_physics_ticks=trainer.physics_ticks,
            added_decisions=trainer.steps-initial_steps,
            elapsed_seconds=time.perf_counter()-started,
            observation_version='gap_geometry_v1',
            action_map=['left', 'right', 'jump', 'double_jump', 'idle'],
            torch_version=torch.__version__,
            training_seed_range=[0, 99999],
            validation_seeds=list(range(settings.get('validation_seed_start', 100000),
                                        settings.get('validation_seed_start', 100000)+settings['validation_episodes']))
        ), indent=2)+'\n')
        return folder/'final.pt'
    finally:
        for env in envs:
            env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--run-dir', type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--timesteps', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--device')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    for option, key in [('timesteps', 'total_timesteps'), ('seed', 'seed'), ('device', 'device')]:
        if getattr(args, option) is not None:
            config[key] = getattr(args, option)
    train(config, args.run_dir, args.resume)
