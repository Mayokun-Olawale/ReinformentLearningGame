"""Record actual deterministic model inference as a GIF and JSON trajectory."""
import argparse
import json
from pathlib import Path
from training_paths import TRAINING_DIR
import pygame
from PIL import Image
from stable_baselines3 import DQN
from game_env import PlatformerGameEnv
from model_profile import action_repeat_for


def record(model_path, folder, action_repeat=None, speed=1, seed=200000):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    model = DQN.load(model_path, device="cpu")
    action_repeat = action_repeat or action_repeat_for(model_path)
    env = PlatformerGameEnv(render_mode="rgb_array", character="VirtualGuy", randomize_obstacles=True)
    obs, _ = env.reset(seed=seed)
    frames, trajectory = [], []
    if speed <= 0:
        raise ValueError("speed must be positive")
    stride = 6 * speed
    try:
        ticks = 0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            for _ in range(action_repeat):
                obs, reward, terminated, truncated, info = env.step(int(action))
                ticks += 1
                player = env.game.player
                trajectory.append(dict(tick=ticks, action=int(action), x=player.rect.x,
                                       y=player.rect.y, fruit=player.fruit_count,
                                       lives=player.life_count))
                if ticks == 1 or ticks % stride == 0 or terminated or truncated:
                    array = env.render()
                    frames.append(Image.fromarray(array).resize((525, 400)).convert("P", palette=Image.Palette.ADAPTIVE))
                if terminated or truncated:
                    break
            if terminated or truncated:
                break
        summary = dict(seed=seed, randomize_obstacles=True, outcome=player.end_reason, ticks=ticks, fruit=player.fruit_count,
                       damage=3-player.life_count, progress_pixels=env.progress_pixels,
                       action_repeat=action_repeat, recording_speed=speed,
                       trajectory=trajectory)
        (folder / "walkthrough.json").write_text(json.dumps(summary))
        frames[0].save(folder / "walkthrough.gif", save_all=True,
                       append_images=frames[1:], duration=100, loop=0)
        print(json.dumps({k:v for k,v in summary.items() if k != "trajectory"}), flush=True)
        return summary
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--folder", required=True)
    parser.add_argument("--action-repeat", type=int, default=None)
    parser.add_argument("--seed", type=int, default=200000)
    parser.add_argument("--speed", type=int, default=1)
    args = parser.parse_args()
    record(args.model, args.folder, args.action_repeat, args.speed, args.seed)
