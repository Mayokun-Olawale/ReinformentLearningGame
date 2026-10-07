"""Reward and decision cadence used by the successful DQN attempt."""
from training_paths import PROJECT_ROOT
import gymnasium as gym
from game_env import PlatformerGameEnv


class CourseTrainingEnv(gym.Wrapper):
    """Repeat decisions four ticks; reward progress and finishing, charge waiting."""
    def __init__(self, render_mode=None, randomize_obstacles=False):
        super().__init__(PlatformerGameEnv(render_mode=render_mode, character="VirtualGuy",
                                           randomize_obstacles=randomize_obstacles))

    def step(self, action):
        total = 0.0
        for _ in range(4):
            obs, _, terminated, truncated, info = self.env.step(action)
            events = info["events"]
            total += info["progress_reward"] * 5 - 0.01
            total += 2 * int(events["picked_fruit"]) - 2 * int(events["took_hit"])
            if events["reached_goal"]:
                total += 100
            elif events["fell_off_map"] or self.env.game.player.life_count <= 0:
                total -= 10
            elif truncated:
                total -= 10
            if terminated or truncated:
                break
        return obs, total, terminated, truncated, info
