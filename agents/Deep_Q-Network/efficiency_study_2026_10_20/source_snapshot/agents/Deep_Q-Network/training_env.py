"""Reward and decision cadence used by the successful DQN attempt."""
from training_paths import PROJECT_ROOT
import gymnasium as gym
from game_env import PlatformerGameEnv


class CourseTrainingEnv(gym.Wrapper):
    """Repeat decisions; reward progress and finishing, charge waiting."""
    def __init__(self, render_mode=None, randomize_obstacles=False,
                 action_repeat=4, fruit_reward=2.0):
        if action_repeat not in (1, 2, 4):
            raise ValueError("action_repeat must be 1, 2, or 4")
        if fruit_reward < 0:
            raise ValueError("fruit_reward must be nonnegative")
        super().__init__(PlatformerGameEnv(render_mode=render_mode, character="VirtualGuy",
                                           randomize_obstacles=randomize_obstacles))
        self.action_repeat = action_repeat
        self.fruit_reward = fruit_reward

    def step(self, action): #demo
        total = 0.0
        # One model decision runs for up to four physics ticks.
        # Stop early if the episode ends, and sum rewards over executed ticks.
        ticks = 0
        for _ in range(self.action_repeat):
            # Ignore the base reward (_); compute the training reward below.
            obs, _, terminated, truncated, info = self.env.step(action)
            ticks += 1
            events = info["events"]
            # Base progress is 0.002/new pixel: x5 gives 0.01/new pixel.
            # Charge 0.01 per tick to discourage waiting. Fruit +2; damage -2.
            total += info["progress_reward"] * 5 - 0.01
            total += self.fruit_reward * int(events["picked_fruit"]) - 2 * int(events["took_hit"])
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
        info = dict(info, decision_ticks=ticks)
        return obs, total, terminated, truncated, info
