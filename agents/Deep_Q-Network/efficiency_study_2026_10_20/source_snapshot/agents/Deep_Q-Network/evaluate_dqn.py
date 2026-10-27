"""Save deterministic policy outcomes on repeatable evaluation seeds."""
import argparse
import json
from pathlib import Path
import gymnasium as gym
import torch

from training_paths import MODEL_DIR, TRAINING_DIR
from stable_baselines3 import DQN
from game_env import PlatformerGameEnv
from model_profile import action_repeat_for, positive_int


class LegacyObservations(gym.ObservationWrapper):
    """Expose the unchanged original inputs for paired baseline evaluation."""

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(-1.0, 1.0, shape=(51,), dtype=env.observation_space.dtype)

    def observation(self, observation):
        return observation[:51]


def evaluate(model_path=MODEL_DIR / "dqn_platformer_best.zip", *, episodes=10,
             seed=200000, randomize_obstacles=True,
             legacy_observations=False,
             output_path=TRAINING_DIR / "evaluation_results.json"): #demo
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    torch.set_num_threads(1)
    model = DQN.load(model_path, device="cpu")
    repeat = action_repeat_for(model_path)
    if repeat in (2, 4):
        from training_env import CourseTrainingEnv
        # Always use the original fruit reward for comparable test scores,
        # including models trained with a larger fruit-collection bonus.
        env = CourseTrainingEnv(randomize_obstacles=randomize_obstacles, action_repeat=repeat)
    else:
        env = PlatformerGameEnv(randomize_obstacles=randomize_obstacles)
    if legacy_observations:
        env = LegacyObservations(env)
    results = []
    try:
        if model.observation_space != env.observation_space or model.action_space != env.action_space:
            raise ValueError("Model spaces do not match; retrain with the current observations")
        # Default test seeds start at 200,000, outside training and validation.
        for episode in range(episodes):
            obs, _ = env.reset(seed=seed + episode)
            reward_sum, damage, ticks = 0.0, 0, 0
            while True:
                # Choose the highest-valued action without random exploration.
                # Evaluation executes the trained policy without updating weights.
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(int(action))
                reward_sum += reward
                damage = 3 - env.unwrapped.game.player.life_count
                ticks = env.unwrapped.game.max_time - env.unwrapped.game.time_remaining
                if terminated or truncated:
                    break
            results.append(dict(seed=seed + episode, reward=reward_sum, ticks=ticks,
                                fruit=env.unwrapped.game.player.fruit_count, damage=damage,
                                progress_pixels=env.unwrapped.progress_pixels,
                                outcome=env.unwrapped.game.player.end_reason))
            if (episode + 1) % 10 == 0:
                wins = sum(row["outcome"] == "completed" for row in results)
                print(f"Evaluated {episode + 1}/{episodes}: {wins} completed", flush=True)
    finally:
        env.close()
    # Completion is the main outcome; fruit, damage, and distance help
    # explain behavior. Reward alone does not prove the agent finished.
    summary = dict(
        completion_rate=sum(row["outcome"] == "completed" for row in results) / episodes,
        mean_reward=sum(row["reward"] for row in results) / episodes,
        mean_fruit=sum(row["fruit"] for row in results) / episodes,
        mean_progress_pixels=sum(row["progress_pixels"] for row in results) / episodes,
        mean_damage=sum(row["damage"] for row in results) / episodes,
        falls=sum(row["outcome"] == "fell" for row in results),
        mean_ticks=sum(row["ticks"] for row in results) / episodes,
        timeouts=sum(row["outcome"] == "time_expired" for row in results),
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(model=str(model_path), randomize_obstacles=randomize_obstacles,
                                    model_decisions=model.num_timesteps,
                                    observation_size=model.observation_space.shape[0],
                                    action_repeat=repeat, seed_start=seed,
                                    evaluation_fruit_reward=2.0 if repeat in (2, 4) else 1.0,
                                    summary=summary, episodes=results), indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Evaluation saved to {path}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(MODEL_DIR / "dqn_platformer_best.zip"))
    parser.add_argument("--episodes", type=positive_int, default=10)
    parser.add_argument("--seed", type=int, default=200000)
    parser.add_argument("--output", type=Path, default=TRAINING_DIR / "evaluation_results.json")
    parser.add_argument("--legacy-observations", action="store_true",
                        help="Expose only the original 51 inputs to an older baseline model")
    parser.add_argument("--fixed-course", action="store_true", help="Diagnostic only: use a fixed layout")
    args = parser.parse_args()
    evaluate(args.model, episodes=args.episodes, seed=args.seed,
             randomize_obstacles=not args.fixed_course, output_path=args.output,
             legacy_observations=args.legacy_observations)
