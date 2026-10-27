"""Exercise actual optimization, checkpointing, logging, and saved inference."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agents" / "Deep_Q-Network"))

import numpy as np
import pygame
import pytest

DQN = pytest.importorskip("stable_baselines3").DQN

from game_env import PlatformerGameEnv
from train_dqn import train


@pytest.mark.parametrize("repeat,curriculum,rate,fruit_bonus",
                         [(4, False, 0.0003, 2), (2, True, 0.0001, 4)])
def test_training_updates_weights_saves_and_loads(tmp_path, repeat, curriculum, rate, fruit_bonus):
    pygame.display.quit()
    model_dir = tmp_path / "models"
    log_dir = tmp_path / "logs"
    path = train(total_timesteps=2048, model_dir=model_dir, log_dir=log_dir,
                 max_episode_steps=64, checkpoint_freq=512, chart_path=tmp_path / "training.png",
                 action_repeat=repeat, curriculum=curriculum, learning_rate=rate, fruit_reward=fruit_bonus)
    assert path.is_file()
    assert (model_dir / "dqn_platformer_512_steps.zip").is_file()
    assert (model_dir / "dqn_platformer_1024_steps.zip").is_file()
    assert (log_dir / "episodes.monitor.csv").is_file()
    model = DQN.load(path, device="cpu")
    assert model.num_timesteps == 2048
    assert model._n_updates > 0
    assert model.learning_rate == rate
    assert np.isclose(model.exploration_rate, 0.1)
    env = PlatformerGameEnv()
    try:
        obs, _ = env.reset(seed=123)
        action, _ = model.predict(obs, deterministic=True)
        assert env.action_space.contains(action)
        env.step(int(action))
        assert not pygame.display.get_init()
    finally:
        env.close()


def test_virtual_guy_appearance_preserves_trained_collision_geometry():
    from Game.sprite_utils import load_sprite_sheets
    original = PlatformerGameEnv()
    virtual = PlatformerGameEnv(character="VirtualGuy")
    sprites = load_sprite_sheets("MainCharacters", "VirtualGuy", 32, 32, True)
    for env in (original, virtual):
        env.reset(seed=123)
    for action in [1, 1, 2, 1, 3, 4]:
        a = original.step(action)
        b = virtual.step(action)
        np.testing.assert_array_equal(a[0], b[0])
        assert a[1:] == b[1:]
        assert original.game.player.mask.count() == virtual.game.player.mask.count()
        assert any(virtual.game.player.visual_sprite is frame
                   for frames in sprites.values() for frame in frames)


def test_chart_file_and_visible_policy_playback(tmp_path):
    from plot_training import plot_training
    from watch_dqn import watch
    source = tmp_path / "episodes.monitor.csv"
    source.write_text('#{"t_start": 0}\nr,l,t\n1,20,0.1\n-2,30,0.2\n')
    chart = plot_training(source, tmp_path / "chart.png")
    assert chart.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    # Exercise the real saved policy, drawing loop, termination, and cleanup.
    env = PlatformerGameEnv()
    model = DQN("MlpPolicy", env, device="cpu", seed=123)
    path = tmp_path / "policy.zip"
    model.save(path)
    env.close()
    watch(path, max_episode_steps=2)
    assert not pygame.display.get_init()


def test_evaluation_reports_completion_fruit_damage_and_falls(tmp_path, monkeypatch):
    import json
    import evaluate_dqn
    env = PlatformerGameEnv()
    model = DQN("MlpPolicy", env, device="cpu", seed=123)
    path = tmp_path / "policy.zip"
    model.save(path)
    env.close()
    monkeypatch.setattr(evaluate_dqn, "PlatformerGameEnv",
                        lambda **kwargs: PlatformerGameEnv(max_episode_steps=2, **kwargs))
    output = tmp_path / "evaluation.json"
    summary = evaluate_dqn.evaluate(path, episodes=2, output_path=output)
    assert summary["completion_rate"] == 0
    assert summary["falls"] == 0
    report = json.loads(output.read_text())
    assert [row["seed"] for row in report["episodes"]] == [200000, 200001]
    assert all(row["ticks"] == 2 and row["outcome"] == "time_expired"
               for row in report["episodes"])


def test_saved_decision_cadence_and_training_wrapper(tmp_path):
    from model_profile import action_repeat_for
    from training_env import CourseTrainingEnv
    (tmp_path / "config.json").write_text('{"action_repeat": 4}')
    assert action_repeat_for(tmp_path / "model.zip") == 4
    env = CourseTrainingEnv()
    env.reset(seed=123)
    obs, reward, terminated, truncated, info = env.step(1)
    assert env.unwrapped.game.time_remaining == 17996
    assert env.unwrapped.progress_pixels == 20
    assert np.isclose(reward, 0.16)
    assert not terminated and not truncated
    env.unwrapped.game.time_remaining = 1
    obs, reward, terminated, truncated, info = env.step(4)
    assert truncated and env.unwrapped.game.time_remaining == 0
    assert np.isclose(reward, -10.01)
    env.close()


def test_legacy_baseline_evaluation_uses_original_inputs(tmp_path):
    from evaluate_dqn import LegacyObservations, evaluate
    modern = PlatformerGameEnv(max_episode_steps=2, randomize_obstacles=True)
    legacy = LegacyObservations(PlatformerGameEnv(max_episode_steps=2, randomize_obstacles=True))
    try:
        new_obs, _ = modern.reset(seed=300000)
        old_obs, _ = legacy.reset(seed=300000)
        assert legacy.observation_space.contains(old_obs)
        np.testing.assert_array_equal(old_obs, new_obs[:51])
        for action in (1, 2):
            new_step = modern.step(action)
            old_step = legacy.step(action)
            np.testing.assert_array_equal(old_step[0], new_step[0][:51])
            assert old_step[1:] == new_step[1:]
        model = DQN("MlpPolicy", legacy, device="cpu", seed=123)
        path = tmp_path / "legacy.zip"
        model.save(path)
    finally:
        modern.close()
        legacy.close()
    # The ordinary evaluator rejects legacy weights unless explicitly requested.
    with pytest.raises(ValueError, match="spaces do not match"):
        evaluate(path, episodes=1, output_path=tmp_path / "unused.json")


def test_randomized_training_advances_seeds_and_avoids_holdouts():
    from train_dqn import TrainingCourses
    env = TrainingCourses()
    env.reset(seed=7)
    first = [tuple(obj.rect) for obj in env.unwrapped.game.level.objects]
    env.reset()
    second = [tuple(obj.rect) for obj in env.unwrapped.game.level.objects]
    assert first != second
    assert env.course_seed == 9
    env.reset(seed=99999)
    assert env.course_seed == 0
    env.reset()
    assert env.course_seed == 1
    env.close()


def test_curriculum_changes_training_difficulty_and_returns_to_full_course():
    from train_dqn import TrainingCourses
    env = TrainingCourses(curriculum=True, total_timesteps=100)
    full = PlatformerGameEnv(randomize_obstacles=True)
    try:
        for decisions, difficulty, gaps, guards in [(0, 0.2, 1, 1), (20, 0.6, 3, 2), (50, 1.0, 5, 4)]:
            env.decisions = decisions
            env.reset(seed=7)
            level = env.unwrapped.game.level
            assert level.difficulty == difficulty
            assert len(level.hazards) == guards
            floor_count = sum(block.rect.top == 704 for block in level.platforms)
            assert floor_count == 54 - gaps
        full.reset(seed=7)
        assert [tuple(b.rect) for b in full.game.level.objects] == [tuple(b.rect) for b in level.objects]
        assert [row['difficulty'] for row in env.stage_history] == [0.2, 0.6, 1.0]
    finally:
        env.close()
        full.close()


def test_two_tick_cadence_and_standardized_evaluation_reward(tmp_path, monkeypatch):
    import json
    import evaluate_dqn
    from training_env import CourseTrainingEnv
    (tmp_path / 'config.json').write_text('{"action_repeat": 2, "fruit_reward": 4}')
    env = CourseTrainingEnv(action_repeat=2, randomize_obstacles=True, fruit_reward=4)
    try:
        env.reset(seed=7)
        _, reward, _, _, info = env.step(1)
        assert info['decision_ticks'] == 2
        assert env.unwrapped.game.time_remaining == 17998
        assert np.isclose(reward, 0.08)
        model = DQN('MlpPolicy', env, device='cpu', seed=7)
        path = tmp_path / 'policy.zip'
        model.save(path)
    finally:
        env.close()
    captured = []
    def make_env(**kwargs):
        result = CourseTrainingEnv(**kwargs)
        captured.append(result)
        result.unwrapped.game.max_time = 2
        return result
    monkeypatch.setattr('training_env.CourseTrainingEnv', make_env)
    evaluate_dqn.evaluate(path, episodes=1, output_path=tmp_path/'evaluation.json')
    assert captured[0].action_repeat == 2
    assert captured[0].fruit_reward == 2
    data = json.loads((tmp_path/'evaluation.json').read_text())
    assert data['episodes'][0]['ticks'] == 2
    assert data['evaluation_fruit_reward'] == 2
