"""Reuse DQN's controls, reward, seed distribution and curriculum verbatim."""
import sys
from pathlib import Path

DQN_DIR = Path(__file__).resolve().parents[1] / 'Deep_Q-Network'
if str(DQN_DIR) not in sys.path:
    sys.path.insert(0, str(DQN_DIR))
from training_env import CourseTrainingEnv
from train_dqn import TrainingCourses


def make_env(config, index=0, training=True, render_mode=None):
    if training:
        env = TrainingCourses(action_repeat=config['action_repeat'],
                              fruit_reward=config['fruit_reward'],
                              curriculum=config['curriculum'],
                              total_timesteps=max(1, config['total_timesteps'] // config['num_envs']))
    else:
        env = CourseTrainingEnv(action_repeat=config['action_repeat'],
                                randomize_obstacles=True, render_mode=render_mode)
    env.unwrapped.game.max_time = config['max_episode_steps']
    return env
