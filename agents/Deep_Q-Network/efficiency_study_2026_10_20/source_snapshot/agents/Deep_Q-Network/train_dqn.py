"""Train DQN on changing courses; validate on disjoint, held-out seeds."""
import argparse
import json
import time
from pathlib import Path
import gymnasium as gym
import torch
from training_paths import MODEL_DIR, LOG_DIR, CHART_PATH
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from training_env import CourseTrainingEnv
from plot_training import plot_training
from model_profile import positive_int


class TrainingCourses(gym.Wrapper):
    def __init__(self, *, action_repeat=4, fruit_reward=2.0, curriculum=False,
                 total_timesteps=500000):
        super().__init__(CourseTrainingEnv(randomize_obstacles=True,
                                         action_repeat=action_repeat, fruit_reward=fruit_reward))
        self.course_seed = 0
        self.curriculum = curriculum
        self.total_timesteps = total_timesteps
        self.decisions = 0
        self.physics_ticks = 0
        self.stage_history = []

    def step(self, action):
        result = self.env.step(action)
        self.decisions += 1
        self.physics_ticks += result[4]["decision_ticks"]
        return result

    def reset(self, *, seed=None, options=None): #demo
        # An explicit seed sets the starting course. Later episode resets
        # advance seeds within the training range 0..99,999.
        if seed is not None:
            self.course_seed = seed % 100000
        course_seed = self.course_seed
        self.course_seed = (self.course_seed + 1) % 100000
        options = dict(options or {})
        fraction = self.decisions / self.total_timesteps
        difficulty = (0.2 if fraction < 0.2 else 0.6 if fraction < 0.5 else 1.0) if self.curriculum else 1.0
        options["difficulty"] = difficulty
        if not self.stage_history or self.stage_history[-1]["difficulty"] != difficulty:
            self.stage_history.append(dict(decisions=self.decisions, physics_ticks=self.physics_ticks,
                                           difficulty=difficulty))
        # Rebuild the course and reset the player, preserving the model learning.
        return self.env.reset(seed=course_seed, options=options)


class Validation(BaseCallback):
    def __init__(self, folder, every=25000, action_repeat=4):
        super().__init__()
        self.action_repeat = action_repeat
        self.folder, self.every, self.best = folder, every, -1
        history_path = self.folder / "validation.json"
        self.history = json.loads(history_path.read_text()) if history_path.exists() else []
        if self.history:
            self.best = max(row["wins"] * 10000 + sum(course["progress"] for course in row["courses"]) / 5
                            for row in self.history)

    def _on_step(self):
        if self.num_timesteps % self.every:
            return True
        env = CourseTrainingEnv(randomize_obstacles=True, action_repeat=self.action_repeat)
        results = []
        try:
            # Validation courses use seeds outside the training range.
            for seed in range(100000,100005):
                obs,_ = env.reset(seed=seed)
                reward_sum = 0.0
                while True:
                    action,_=self.model.predict(obs,deterministic=True)
                    obs,reward,terminated,truncated,_=env.step(int(action))
                    reward_sum += reward
                    if terminated or truncated:break
                results.append(dict(seed=seed,outcome=env.unwrapped.game.player.end_reason,
                                    progress=env.unwrapped.progress_pixels, reward=reward_sum))
        finally:
            env.close()
        wins=sum(row['outcome']=='completed' for row in results)
        # Rank checkpoints by completions and average forward progress.
        score=wins*10000+sum(row['progress'] for row in results)/5
        self.history.append(dict(steps=self.num_timesteps,wins=wins,courses=results))
        (self.folder/'validation.json').write_text(json.dumps(self.history,indent=2))
        if score>self.best:
            self.best=score
            self.model.save(self.folder/'dqn_platformer_best.zip')
        print(f"Validation at {self.num_timesteps}: {wins}/5 completed",flush=True)
        return True


def train(randomize_obstacles=True, *, total_timesteps=500000, seed=7,
          log_dir=LOG_DIR, model_dir=MODEL_DIR, device='cpu',
          max_episode_steps=18000, checkpoint_freq=25000, chart_path=CHART_PATH, resume=None,
          action_repeat=4, learning_rate=0.0003, curriculum=False, fruit_reward=2.0,
          validation_freq=25000): #demo
    if not randomize_obstacles:
        raise ValueError('Training requires randomized courses')
    if total_timesteps<=0 or checkpoint_freq<=0 or validation_freq<=0:
        raise ValueError('Timesteps and checkpoint frequency must be positive')
    if not 0 < learning_rate < float('inf'):
        raise ValueError('Learning rate must be positive and finite')
    torch.set_num_threads(1)
    log_dir,model_dir=Path(log_dir),Path(model_dir)
    log_dir.mkdir(parents=True,exist_ok=True);model_dir.mkdir(parents=True,exist_ok=True)
    courses=TrainingCourses(action_repeat=action_repeat, fruit_reward=fruit_reward,
                            curriculum=curriculum, total_timesteps=total_timesteps)
    env=courses
    env.unwrapped.game.max_time=max_episode_steps
    env=Monitor(env,filename=str(log_dir/'episodes'),override_existing=resume is None)
    config=dict(randomize_obstacles=True,action_repeat=action_repeat,seed=seed,steps=total_timesteps,
                learning_rate=learning_rate, curriculum=curriculum, fruit_reward=fruit_reward,
                nominal_physics_tick_budget=total_timesteps*action_repeat,
                validation_frequency_decisions=validation_freq,
                checkpoint_selection="validation completions, then mean progress",
                observation_size=env.observation_space.shape[0],observation_version="gap_geometry_v1",
                training_seeds=[0,99999],validation_seeds=[100000,100004],
                exploration_fraction=0.8,exploration_final_eps=0.1)
    (model_dir/'config.json').write_text(json.dumps(config,indent=2))
    try:
        # Stable-Baselines3 supplies DQN learning and experience replay.
        # MlpPolicy maps 60 observations through two 128-neuron hidden layers
        # to five Q-values estimating discounted future reward per action.
        # Exploration starts random and declines to a 10% random-action rate.
        model=DQN('MlpPolicy',env,learning_rate=learning_rate,buffer_size=100000,
                  learning_starts=2000,batch_size=128,gamma=0.995,
                  train_freq=4,gradient_steps=1,target_update_interval=500,
                  exploration_fraction=0.8,exploration_initial_eps=1,
                  exploration_final_eps=0.1,policy_kwargs=dict(net_arch=[128,128]),
                  seed=seed,device=device,verbose=0)
        if resume is not None:
            model = DQN.load(resume, env=env, device=device)
            config["resumed_from_decisions"] = model.num_timesteps
            model.learning_starts = model.num_timesteps + 2000
            (model_dir / "config.json").write_text(json.dumps(config, indent=2))
        callbacks=[Validation(model_dir, every=validation_freq, action_repeat=action_repeat)]
        # Explicit short verification runs can request periodic checkpoints.
        if total_timesteps<25000:
            callbacks.append(CheckpointCallback(checkpoint_freq,str(model_dir),'dqn_platformer'))
        print(f'Training {total_timesteps} decisions on randomized courses',flush=True)
        # Collect transitions into replay memory and train on sampled batches.
        # A timestep here is a model decision, containing up to four game ticks.
        started = time.monotonic()
        model.learn(total_timesteps=total_timesteps,callback=CallbackList(callbacks),
                    reset_num_timesteps=resume is None)
        path=model_dir/'dqn_platformer_final.zip';model.save(path)
        config.update(final_model_decisions=model.num_timesteps,
                      actual_physics_ticks=courses.physics_ticks,
                      training_elapsed_seconds=time.monotonic()-started,
                      curriculum_stages=courses.stage_history)
        (model_dir/'config.json').write_text(json.dumps(config,indent=2))
        plot_training(log_dir/'episodes.monitor.csv',chart_path,step_unit='decisions')
        return path
    finally:
        env.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--timesteps',type=positive_int,default=500000)
    parser.add_argument('--seed',type=int,default=7)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--model-dir', type=Path, default=MODEL_DIR)
    parser.add_argument('--log-dir', type=Path, default=LOG_DIR)
    parser.add_argument('--chart-path', type=Path, default=CHART_PATH)
    parser.add_argument('--action-repeat', type=int, choices=(1, 2, 4), default=4)
    parser.add_argument('--learning-rate', type=float, default=0.0003)
    parser.add_argument('--curriculum', action='store_true')
    parser.add_argument('--fruit-reward', type=float, default=2.0)
    parser.add_argument('--validation-freq', type=positive_int, default=25000)
    args=parser.parse_args()
    train(total_timesteps=args.timesteps,seed=args.seed,resume=args.resume,
          model_dir=args.model_dir,log_dir=args.log_dir,chart_path=args.chart_path,
          action_repeat=args.action_repeat,learning_rate=args.learning_rate,
          curriculum=args.curriculum,fruit_reward=args.fruit_reward,
          validation_freq=args.validation_freq)
