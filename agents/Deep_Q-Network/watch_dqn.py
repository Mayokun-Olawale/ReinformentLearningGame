"""Watch the saved DQN policy play as VirtualGuy; Escape or close exits."""
import argparse
from training_paths import MODEL_DIR, TRAINING_DIR
import pygame
from stable_baselines3 import DQN
from game_env import PlatformerGameEnv
from model_profile import action_repeat_for, positive_int


def watch(model_path=MODEL_DIR / "dqn_platformer_best.zip", *, episodes=1,
          seed=200000, randomize_obstacles=True, max_episode_steps=18000):
    model = DQN.load(model_path, device="cpu")
    repeat = action_repeat_for(model_path)
    env = PlatformerGameEnv(render_mode="human", randomize_obstacles=randomize_obstacles,
                            max_episode_steps=max_episode_steps, character="VirtualGuy")
    try:
        if model.action_space != env.action_space or model.observation_space != env.observation_space:
            raise ValueError("Saved model spaces do not match the game environment. Retrain with the 51-value observations.")
        for episode in range(episodes):
            obs, _ = env.reset(seed=seed + episode)
            pygame.display.set_caption(f"VirtualGuy — {model_path} | N: next model | Esc: exit")
            print(f"Playing {model_path}. N skips; Escape exits.", flush=True)
            total_reward = 0.0
            while True:
                events = list(pygame.event.get())
                if any(event.type == pygame.QUIT or
                       (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE)
                       for event in events):
                    return False
                if any(event.type == pygame.KEYDOWN and event.key == pygame.K_n for event in events):
                    break
                action, _ = model.predict(obs, deterministic=True)
                for _ in range(repeat):
                    obs, reward, terminated, truncated, _ = env.step(int(action))
                    total_reward += reward
                    if terminated or truncated:
                        break
                if terminated or truncated:
                    print(f"Episode {episode + 1}: {env.game.player.end_reason}, "
                          f"reward={total_reward:.1f}, fruit={env.game.player.fruit_count}")
                    break
        return True
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(MODEL_DIR / "dqn_platformer_best.zip"))
    parser.add_argument("--episodes", type=positive_int, default=1)
    parser.add_argument("--seed", type=int, default=200000)
    parser.add_argument("--fixed-course", action="store_true", help="Diagnostic only: use a fixed layout")
    args = parser.parse_args()
    watch(args.model, episodes=args.episodes, seed=args.seed,
          randomize_obstacles=not args.fixed_course)
