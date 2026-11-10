"""Numerical boundaries, clipping, and real-game optimization/checkpoint tests."""
import json
import numpy as np
import pytest
import torch
from agents.Proximal_Policy_Optimization.ppo.buffer import RolloutBuffer
from agents.Proximal_Policy_Optimization.ppo.trainer import ppo_loss
from agents.Proximal_Policy_Optimization.train_ppo import train, validate, DEFAULT_CONFIG
from agents.Proximal_Policy_Optimization.evaluate import evaluate, load_policy


@pytest.mark.parametrize('terminated,truncated,expected', [
    (True, False, [1., 2.]), (False, True, [10., 2.]),
    (False, False, [11.71, 2.])])
def test_gae_boundaries(terminated, truncated, expected):
    buffer = RolloutBuffer()
    buffer.add([[0.]], [0], [0.], [0.], [1.], [terminated], [truncated], [10.])
    buffer.add([[0.]], [0], [0.], [0.], [2.], [True], [False], [99.])
    advantages, returns = buffer.compute_gae(.9, .95)
    torch.testing.assert_close(advantages[:, 0], torch.tensor(expected))
    torch.testing.assert_close(returns, advantages)
    list(buffer.minibatches(2, 1))
    with pytest.raises(RuntimeError, match='consumed'):
        list(buffer.minibatches(2, 1))


def test_policy_and_old_value_clipping():
    loss, stats = ppo_loss(torch.tensor([np.log(2), np.log(.5)]), torch.zeros(2),
                           torch.tensor([1., -1.]), torch.zeros(2), torch.zeros(2),
                           torch.zeros(2), torch.zeros(2), value_coef=0, entropy_coef=0)
    assert float(loss) == pytest.approx(-.2)
    assert float(stats['clip_fraction']) == 1
    loss, _ = ppo_loss(torch.zeros(1), torch.zeros(1), torch.zeros(1),
                       torch.tensor([2.]), torch.tensor([0.]), torch.tensor([2.]),
                       torch.zeros(1), value_coef=1, entropy_coef=0, value_clip=.2)
    assert float(loss) == pytest.approx(3.24)


def test_real_training_and_saved_inference(tmp_path):
    config = dict(total_timesteps=128, num_envs=2, rollout_steps=16, minibatch_size=16,
                  max_episode_steps=24, validation_freq=64, validation_episodes=1)
    # Repeat seed gives identical initialization, making weight-change check meaningful.
    torch.set_num_threads(1)
    torch.manual_seed(7)
    from agents.Proximal_Policy_Optimization.ppo.network import ActorCritic
    initial = ActorCritic(60, 5).actor.weight.detach().clone()
    path = train(config, tmp_path/'run')
    model, settings, steps = load_policy(path)
    assert steps == 128
    assert not torch.equal(initial, model.actor.weight)
    assert all(torch.isfinite(p).all() for p in model.parameters())
    assert (path.parent/'best.pt').exists()
    report = evaluate(path, seeds=[200000, 200001], output=tmp_path/'evaluation.json')
    assert len(report['episodes']) == 2
    assert all(r['ticks'] == 24 for r in report['episodes'])
    assert report['summary']['timeouts'] == 2
    assert (path.parent/'metrics.csv').read_text().count('\n') == 5
    assert json.loads((path.parent/'episodes.json').read_text())
    resumed = train(dict(config, total_timesteps=32), tmp_path/'resumed', resume=path)
    assert load_policy(resumed)[2] == 160


def test_collection_bootstraps_final_observation_not_reset():
    import gymnasium as gym
    from agents.Proximal_Policy_Optimization.ppo.trainer import PPOTrainer
    class TimeoutEnv(gym.Env):
        observation_space = gym.spaces.Box(-10, 10, (1,), dtype=np.float32)
        action_space = gym.spaces.Discrete(2)
        def reset(self, **kwargs):
            return np.array([-5.], dtype=np.float32), {}
        def step(self, action):
            return np.array([3.], dtype=np.float32), 1., False, True, {'decision_ticks': 1}
        @property
        def game(self):
            from types import SimpleNamespace
            return SimpleNamespace(player=SimpleNamespace(fruit_count=0, end_reason='time_expired'))
        progress_pixels = 0
    config = json.loads(DEFAULT_CONFIG.read_text())
    trainer = PPOTrainer([TimeoutEnv()], config)
    class KnownValue(torch.nn.Module):
        def distribution(self, obs):
            return torch.distributions.Categorical(logits=torch.zeros(len(obs), 2)), obs[:, 0]
        def forward(self, obs):
            return torch.zeros(len(obs), 2), obs[:, 0]
    trainer.model = KnownValue()
    buffer = trainer.collect_rollout(2)
    torch.testing.assert_close(buffer.next_values, torch.full((2, 1), 3.))
    torch.testing.assert_close(buffer.returns, torch.full((2, 1), 3.97))
    assert trainer.obs[0, 0] == -5


def test_invalid_config():
    config = json.loads(DEFAULT_CONFIG.read_text())
    with pytest.raises(ValueError, match='divisible'):
        validate(dict(config, total_timesteps=3))


def test_shared_reward_components_and_seed_determinism():
    from agents.Proximal_Policy_Optimization.envs import make_env
    config = json.loads(DEFAULT_CONFIG.read_text())
    a, b = make_env(config), make_env(config)
    try:
        np.testing.assert_array_equal(a.reset(seed=7)[0], b.reset(seed=7)[0])
        for action in (1, 2, 1, 3, 4):
            left, right = a.step(action), b.step(action)
            np.testing.assert_array_equal(left[0], right[0])
            assert left[1:] == right[1:]
            assert sum(left[4]['reward_components'].values()) == pytest.approx(left[1])
    finally:
        a.close()
        b.close()
