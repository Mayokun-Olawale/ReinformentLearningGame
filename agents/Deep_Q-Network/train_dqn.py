"""Train DQN on changing courses; validate on disjoint, held-out seeds."""
import argparse
import json
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
    def __init__(self):
        super().__init__(CourseTrainingEnv(randomize_obstacles=True))
        self.course_seed = 0

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.course_seed = seed % 100000
        course_seed = self.course_seed
        self.course_seed = (self.course_seed + 1) % 100000
        return self.env.reset(seed=course_seed, options=options)


class Validation(BaseCallback):
    def __init__(self, folder, every=25000):
        super().__init__()
        self.folder, self.every, self.best = folder, every, -1
        history_path = self.folder / "validation.json"
        self.history = json.loads(history_path.read_text()) if history_path.exists() else []
        if self.history:
            self.best = max(row["wins"] * 10000 + sum(course["progress"] for course in row["courses"]) / 5
                            for row in self.history)

    def _on_step(self):
        if self.num_timesteps % self.every:
            return True
        env = CourseTrainingEnv(randomize_obstacles=True)
        results = []
        try:
            for seed in range(100000,100005):
                obs,_ = env.reset(seed=seed)
                while True:
                    action,_=self.model.predict(obs,deterministic=True)
                    obs,_,terminated,truncated,_=env.step(int(action))
                    if terminated or truncated:break
                results.append(dict(seed=seed,outcome=env.unwrapped.game.player.end_reason,
                                    progress=env.unwrapped.progress_pixels))
        finally:
            env.close()
        wins=sum(row['outcome']=='completed' for row in results)
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
          max_episode_steps=18000, checkpoint_freq=25000, chart_path=CHART_PATH, resume=None):
    if not randomize_obstacles:
        raise ValueError('Training requires randomized courses')
    if total_timesteps<=0 or checkpoint_freq<=0:
        raise ValueError('Timesteps and checkpoint frequency must be positive')
    torch.set_num_threads(1)
    log_dir,model_dir=Path(log_dir),Path(model_dir)
    log_dir.mkdir(parents=True,exist_ok=True);model_dir.mkdir(parents=True,exist_ok=True)
    env=TrainingCourses()
    env.unwrapped.game.max_time=max_episode_steps
    env=Monitor(env,filename=str(log_dir/'episodes'),override_existing=resume is None)
    config=dict(randomize_obstacles=True,action_repeat=4,seed=seed,steps=total_timesteps,
                training_seeds=[0,99999],validation_seeds=[100000,100004],
                exploration_fraction=0.8,exploration_final_eps=0.1)
    (model_dir/'config.json').write_text(json.dumps(config,indent=2))
    try:
        model=DQN('MlpPolicy',env,learning_rate=0.0003,buffer_size=100000,
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
        callbacks=[Validation(model_dir)]
        # Explicit short verification runs can request periodic checkpoints.
        if total_timesteps<25000:
            callbacks.append(CheckpointCallback(checkpoint_freq,str(model_dir),'dqn_platformer'))
        print(f'Training {total_timesteps} decisions on randomized courses',flush=True)
        model.learn(total_timesteps=total_timesteps,callback=CallbackList(callbacks),
                    reset_num_timesteps=resume is None)
        path=model_dir/'dqn_platformer_final.zip';model.save(path)
        plot_training(log_dir/'episodes.monitor.csv',chart_path,step_unit='decisions')
        return path
    finally:
        env.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--timesteps',type=positive_int,default=500000)
    parser.add_argument('--seed',type=int,default=7)
    parser.add_argument('--resume', type=Path)
    args=parser.parse_args()
    train(total_timesteps=args.timesteps,seed=args.seed,resume=args.resume)
