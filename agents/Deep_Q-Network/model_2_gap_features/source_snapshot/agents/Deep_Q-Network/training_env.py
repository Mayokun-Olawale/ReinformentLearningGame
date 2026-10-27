"""Reward and decision cadence used by the successful DQN attempt."""
from training_paths import PROJECT_ROOT
import gymnasium as gym
from game_env import PlatformerGameEnv


class CourseTrainingEnv(gym.Wrapper):
    """Repeat decisions four ticks; reward progress and finishing, charge waiting."""
    def __init__(self, render_mode=None, randomize_obstacles=False):
        super().__init__(PlatformerGameEnv(render_mode=render_mode, character="VirtualGuy",
                                           randomize_obstacles=randomize_obstacles))

    def step(self, action): #demo
        total = 0.0
        # One model decision runs for up to four physics ticks.
        # Stop early if the episode ends, and sum rewards over executed ticks.
        for _ in range(4):
            # Ignore the base reward (_); compute the training reward below.
            obs, _, terminated, truncated, info = self.env.step(action)
            events = info["events"]
            # Base progress is 0.002/new pixel: x5 gives 0.01/new pixel.
            # Charge 0.01 per tick to discourage waiting. Fruit +2; damage -2.
            total += info["progress_reward"] * 5 - 0.01
            total += 2 * int(events["picked_fruit"]) - 2 * int(events["took_hit"])
            # Finish +100; falling, losing all lives, or timing out costs 10.
            if events["reached_goal"]:
                total += 100
            elif events["fell_off_map"] or self.env.game.player.life_count <= 0:
                total -= 10
            elif truncated:
                total -= 10
            if terminated or truncated:
                break
        # Return the final observation/info and the accumulated decision reward.
        return obs, total, terminated, truncated, info
