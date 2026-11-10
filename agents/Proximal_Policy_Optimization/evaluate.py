"""Evaluate PPO, DQN, random or rightward policies on identical game seeds."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from agents.Proximal_Policy_Optimization.envs import make_env
from agents.Proximal_Policy_Optimization.ppo.network import ActorCritic


def load_policy(path, device='cpu'):
    data = torch.load(path, map_location=device, weights_only=False)
    model = ActorCritic(60, 5, data['config']['hidden_size']).to(device)
    model.load_state_dict(data['model'])
    model.eval()
    return model, data['config'], data['steps']


def evaluate_policy(policy, config, seeds, render=False):
    if not seeds:
        raise ValueError('At least one evaluation seed required')
    env = make_env(config, training=False, render_mode='human' if render else None)
    rows = []
    started = time.perf_counter()
    try:
        for seed in seeds:
            obs, _ = env.reset(seed=int(seed))
            reward_sum = 0.0
            decisions = 0
            while True:
                obs, reward, terminated, truncated, info = env.step(int(policy(obs)))
                reward_sum += reward
                decisions += 1
                if terminated or truncated:
                    break
            game = env.unwrapped.game
            rows.append(dict(seed=int(seed), reward=reward_sum, decisions=decisions,
                             ticks=game.max_time-game.time_remaining,
                             fruit=game.player.fruit_count,
                             fruit_rate=game.player.fruit_count / max(1, len(game.level.fruits)),
                             damage=3-game.player.life_count, progress_pixels=env.unwrapped.progress_pixels,
                             outcome=game.player.end_reason))
    finally:
        env.close()
    summary = {f'mean_{key}': float(np.mean([r[key] for r in rows]))
               for key in ('reward', 'decisions', 'ticks', 'fruit', 'fruit_rate', 'damage', 'progress_pixels')}
    summary.update(completion_rate=sum(r['outcome']=='completed' for r in rows)/len(rows),
                   falls=sum(r['outcome']=='fell' for r in rows),
                   timeouts=sum(r['outcome']=='time_expired' for r in rows),
                   elapsed_seconds=time.perf_counter()-started)
    return dict(summary=summary, episodes=rows)


def evaluate(checkpoint=None, *, algo='ppo', seeds=None, output=None, render=False,
             action_repeat=None, max_episode_steps=None):
    torch.set_num_threads(1)
    seeds = list(range(200000, 200010)) if seeds is None else list(seeds)
    if algo == 'ppo':
        model, config, steps = load_policy(checkpoint)
        def policy(obs):
            with torch.no_grad():
                return model(torch.as_tensor(obs).unsqueeze(0))[0].argmax(-1).item()
    elif algo == 'dqn':
        from stable_baselines3 import DQN
        from model_profile import action_repeat_for
        model = DQN.load(checkpoint, device='cpu')
        if model.observation_space.shape != (60,):
            raise ValueError('Comparison requires the current 60-feature DQN')
        config = dict(action_repeat=action_repeat_for(checkpoint), max_episode_steps=18000)
        steps = model.num_timesteps
        policy = lambda obs: model.predict(obs, deterministic=True)[0]
    else:
        config = dict(action_repeat=4, max_episode_steps=18000)
        rng = np.random.default_rng(7)
        steps = 0
        policy = (lambda obs: rng.integers(5)) if algo == 'random' else (lambda obs: 1)
    config = dict(config)
    if action_repeat is not None:
        config['action_repeat'] = action_repeat
    if max_episode_steps is not None:
        config['max_episode_steps'] = max_episode_steps
    report = evaluate_policy(policy, config, seeds, render)
    report.update(algo=algo, checkpoint=str(checkpoint), model_decisions=steps,
                  action_repeat=config['action_repeat'], evaluation_fruit_reward=2.0,
                  observation_version='gap_geometry_v1', max_episode_steps=config['max_episode_steps'])
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--algo', choices=['ppo', 'dqn', 'random', 'right'], default='ppo')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--seed-list', type=Path, help='JSON list of held-out seeds')
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent / 'runs' / 'evaluation.json')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--action-repeat', type=int, choices=[1, 2, 4])
    parser.add_argument('--max-episode-steps', type=int)
    args = parser.parse_args()
    if args.algo in ('ppo', 'dqn') and args.checkpoint is None:
        parser.error('--checkpoint is required for a learned policy')
    seeds = json.loads(args.seed_list.read_text()) if args.seed_list else list(range(200000, 200000+args.episodes))
    print(json.dumps(evaluate(args.checkpoint, algo=args.algo, seeds=seeds, output=args.output,
                             render=args.render, action_repeat=args.action_repeat,
                             max_episode_steps=args.max_episode_steps)['summary'], indent=2))
