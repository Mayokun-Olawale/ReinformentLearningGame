"""Synchronous batched collection and PPO-Clip optimization."""
import random
import time
from pathlib import Path
import numpy as np
import torch
from .network import ActorCritic
from .buffer import RolloutBuffer


def ppo_loss(logp, old_logp, advantages, values, old_values, returns, entropy,
             clip=0.2, value_coef=0.5, entropy_coef=0.01, value_clip=None):
    log_ratio = logp - old_logp
    ratio = log_ratio.exp()
    policy = -torch.minimum(ratio * advantages, ratio.clamp(1-clip, 1+clip) * advantages).mean()
    error = (values - returns).square()
    if value_clip is not None:
        clipped = old_values + (values - old_values).clamp(-value_clip, value_clip)
        error = torch.maximum(error, (clipped - returns).square())
    value = error.mean()
    loss = policy + value_coef * value - entropy_coef * entropy.mean()
    return loss, dict(policy_loss=policy.detach(), value_loss=value.detach(),
                      entropy=entropy.mean().detach(),
                      approx_kl=((ratio - 1) - log_ratio).mean().detach(),
                      clip_fraction=((ratio - 1).abs() > clip).float().mean().detach())


class PPOTrainer:
    def __init__(self, envs, config):
        self.envs, self.config = envs, config
        self.device = torch.device(config['device'])
        random.seed(config['seed'])
        np.random.seed(config['seed'])
        torch.manual_seed(config['seed'])
        self.model = ActorCritic(envs[0].observation_space.shape[0], envs[0].action_space.n,
                                 config['hidden_size']).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['learning_rate'], eps=1e-5)
        self.steps = self.physics_ticks = 0
        self.obs = np.stack([env.reset(seed=(config['seed'] + i * 1000) % 100000)[0]
                             for i, env in enumerate(envs)])
        self.episode_returns = np.zeros(len(envs))
        self.episode_lengths = np.zeros(len(envs), dtype=int)
        self.episodes = []
        self.reward_components = [{} for _ in envs]

    def tensor(self, value):
        return torch.as_tensor(value, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def collect_rollout(self, steps_per_env):
        buffer = RolloutBuffer()
        for _ in range(steps_per_env):
            observations = self.tensor(self.obs)
            distribution, values = self.model.distribution(observations)
            actions = distribution.sample()
            results = [env.step(int(action)) for env, action in zip(self.envs, actions.cpu().numpy())]
            next_obs, rewards, terminated, truncated, infos = zip(*results)
            # Evaluate actual final observations BEFORE any reset.
            _, next_values = self.model(self.tensor(np.stack(next_obs)))
            buffer.add(observations, actions, distribution.log_prob(actions), values,
                       self.tensor(rewards), torch.tensor(terminated, device=self.device),
                       torch.tensor(truncated, device=self.device), next_values)
            self.obs = np.stack(next_obs)
            self.steps += len(self.envs)
            self.physics_ticks += sum(info['decision_ticks'] for info in infos)
            self.episode_returns += rewards
            self.episode_lengths += 1
            for i, env in enumerate(self.envs):
                for key, value in infos[i].get('reward_components', {}).items():
                    self.reward_components[i][key] = self.reward_components[i].get(key, 0.0) + value
                if terminated[i] or truncated[i]:
                    game = env.unwrapped.game
                    self.episodes.append(dict(steps=self.steps, reward=float(self.episode_returns[i]),
                                              decisions=int(self.episode_lengths[i]),
                                              fruit=game.player.fruit_count,
                                              outcome=game.player.end_reason,
                                              reward_components=self.reward_components[i],
                                              progress=env.unwrapped.progress_pixels))
                    self.reward_components[i] = {}
                    self.obs[i] = env.reset()[0]
                    self.episode_returns[i] = self.episode_lengths[i] = 0
        buffer.compute_gae(self.config['gamma'], self.config['gae_lambda'])
        return buffer

    def update(self, rollout, entropy_coef=None):
        entropy_coef = self.config['entropy_coef'] if entropy_coef is None else entropy_coef
        metrics = []
        for obs, actions, old_logp, old_values, advantages, returns in rollout.minibatches(
                self.config['minibatch_size'], self.config['epochs']):
            distribution, values = self.model.distribution(obs)
            loss, stats = ppo_loss(distribution.log_prob(actions), old_logp, advantages,
                                   values, old_values, returns, distribution.entropy(),
                                   self.config['clip'], self.config['value_coef'],
                                   entropy_coef, self.config['value_clip'])
            # Guard the proposed policy shift before another optimizer step.
            if self.config['target_kl'] and stats['approx_kl'] > self.config['target_kl']:
                break
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite PPO loss')
            self.optimizer.zero_grad()
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['max_grad_norm'])
            self.optimizer.step()
            metrics.append({**{k: float(v) for k, v in stats.items()}, 'grad_norm': float(grad)})
        return {key: float(np.mean([row[key] for row in metrics])) for key in metrics[0]} if metrics else {}

    def save_checkpoint(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(model=self.model.state_dict(), optimizer=self.optimizer.state_dict(),
                        config=self.config, steps=self.steps, physics_ticks=self.physics_ticks,
                        torch_rng=torch.get_rng_state()), path)

    @classmethod
    def load(cls, path, envs, device='cpu'):
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        config = dict(checkpoint['config'], device=device)
        trainer = cls(envs, config)
        trainer.model.load_state_dict(checkpoint['model'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer'])
        trainer.steps, trainer.physics_ticks = checkpoint['steps'], checkpoint['physics_ticks']
        torch.set_rng_state(checkpoint['torch_rng'].cpu())
        return trainer
