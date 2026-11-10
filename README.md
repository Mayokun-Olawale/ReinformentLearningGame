# ReinformentLearningGame

The Gymnasium environment uses the game's original sprite player, mask collisions,
scrolling course, gaps, patrolling enemies, fruit, and trophy. Human gameplay uses
that same engine. Both human input and agents advance the same physics engine.

Install dependencies with `python -m pip install -r requirements.txt`.
Launch human gameplay from the repository root with `python -m Game.main`.

```python
from game_env import PlatformerGameEnv

env = PlatformerGameEnv(render_mode=None, randomize_obstacles=False)
obs, info = env.reset(seed=123)
while True:
    action = env.action_space.sample()  # Replace with your policy's action.
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
env.close()
```

Actions are `0` left, `1` right, `2` jump, `3` double jump, and `4` idle.
Jump actions preserve horizontal momentum. Observations contain 60 normalized
`float32` values, clipped to [-1, 1]:

| Indices | Features |
| --- | --- |
| 0–5 | Player x/y, velocity x/y, lives, remaining time |
| 6–7 | Grounded and double-jump availability, separately |
| 8–16 | Three nearest uncollected fruits: relative x/y and presence |
| 17–24 | Two nearest guards: relative x/y, patrol velocity and presence |
| 25–48 | Eight terrain samples: floor below, nearest platform height and presence |
| 49–50 | Relative goal x/y |
| 51–55 | Next floor gap: presence, signed edge distance, width, landing distance, landing height |
| 56–59 | Next raised platform: presence, leading-edge distance, height, width |

Terrain is sampled at -384, -192, -96, -32, +32, +96, +192 and +384 pixels
from the player's center. Platform heights are relative to the player's feet.
Object offsets are relative to the player, scaled by viewport width/height;
goal x is scaled by course width. Missing object slots are zero with presence=0.
The nearest objects are ordered by distance. These observations expose geometry
without image processing, but terrain samples remain an approximation of the map.

New training runs use the current 60-value observation format. Older 51-input
models can be evaluated and played with `--legacy-observations`; do not resume
them in the 60-input training environment.

Gap and platform distances/widths are scaled by viewport width; heights are
relative to the player’s feet and scaled by viewport height. The next gap is
the first internal floor gap whose far edge is ahead of the player’s center,
including a gap currently underneath. Its edge distance is negative while
inside it. The next raised platform has the nearest leading edge at or ahead
of the player’s center. Missing features are zero with presence=0. These
features describe geometry rather than guaranteed jump reachability.

Base event rewards remain +1 for collecting fruit, -1 for taking damage, +2 for the trophy,
and -2 for falling or timing out. Goal, falling, and zero lives terminate an
episode; timeout truncates it. Each new furthest x position also earns +0.002
per pixel; backtracking does not earn the bonus again. The default limit is 18,000 ticks (300 seconds at
60 FPS), configurable through `max_episode_steps`.

`render_mode=None` runs without initializing a display or limiting FPS.
`render_mode="human"` shows the original assets with a scrolling camera;
`render_mode="rgb_array"` returns an `(800, 1050, 3)` RGB array without a window.
Fixed courses use a deterministic layout; `randomize_obstacles=True` regenerates
the original course's gaps, floating blocks, and fruit using the reset seed.
Call `reset()` before stepping and after each episode ends.

To verify the environment, install pytest and run
`SDL_VIDEODRIVER=dummy python -m pytest -q`.

Training tools are in `agents/Deep_Q-Network/`. Training and playback default to
randomized courses; each training episode receives a different seeded layout.
Install training dependencies and start a new run:

```bash
python -m pip install -r agents/Deep_Q-Network/requirements-training.txt
python agents/Deep_Q-Network/train_dqn.py --timesteps 500000
```

For the gap-feature experiment, save a fresh run separately to preserve existing
baseline models and logs (run from the repository root):

```bash
python agents/Deep_Q-Network/train_dqn.py --timesteps 500000 --seed 7 --model-dir agents/Deep_Q-Network/gap_experiment/models --log-dir agents/Deep_Q-Network/gap_experiment/logs --chart-path agents/Deep_Q-Network/gap_experiment/training.png
python agents/Deep_Q-Network/evaluate_dqn.py --model agents/Deep_Q-Network/gap_experiment/models/dqn_platformer_best.zip --episodes 100 --seed 300000 --output agents/Deep_Q-Network/gap_experiment/evaluation.json
```

The evaluation command saves results separately. Compare completion and fruit collection on
identical held-out seeds. The evaluator can expose the unchanged first 51 inputs
to an older baseline with `--legacy-observations`:

```bash
python agents/Deep_Q-Network/evaluate_dqn.py --model agents/Deep_Q-Network/saved_models/dqn_platformer_best.zip --legacy-observations --episodes 100 --seed 300000 --output agents/Deep_Q-Network/gap_experiment/baseline_evaluation.json
```
The observation change alone does not establish improved performance.

The pipeline uses the original course generator, varying gaps, floating platforms,
fruit and guard placement. It does not randomize gravity or change game physics.
Training uses course seeds 0–99,999. Every 25,000 decisions it validates on five
held-out seeds 100,000–100,004. The best validation checkpoint and final model are
saved in `saved_models/`, with configuration and validation metrics; no fixed-course
or demonstration models are retained. Training logs remain in `tensorboard_logs/`.

Training uses four-tick actions and rewards +0.01 per pixel of new forward progress,
-0.01 per tick, +2 fruit, -2 damage, +100 trophy, and -10 fall/defeat/timeout.
Exploration decays over 80% of training to 0.1. The final chart is `training_results.png`.

Evaluate on twenty fresh course seeds, independent of training and validation:

```bash
python agents/Deep_Q-Network/evaluate_dqn.py --episodes 20
```

The default seeds start at 200,000. `evaluation_results.json` includes completion,
fruit, damage, falls and progress. Reported success rates describe that test set,
not a guarantee of success on every generated course. Playback uses the
validation-selected best checkpoint.

Watch the best randomized-course policy as VirtualGuy:

```bash
python agents/Deep_Q-Network/watch_dqn.py --episodes 3
```

Each episode uses a different layout. Escape closes playback; N skips an episode.
Use `--fixed-course` only for a diagnostic fixed-layout run. Playback automatically
loads the saved action cadence from `saved_models/config.json`.

`record_walkthrough.py` can save a randomized-course GIF and full trajectory.
`plot_training.py` can regenerate the chart from the episode log. The reusable
helpers are `training_env.py`, `training_paths.py`, and `model_profile.py`.

The second model and its results are archived separately in
[`agents/Deep_Q-Network/model_2_gap_features/`](agents/Deep_Q-Network/model_2_gap_features/README.md),
including selected/final checkpoints, evaluation results, training logs, charts,
and a gameplay GIF with its trajectory. The first model's original files remain
in their existing locations. Model checkpoints, logs, and recordings are organized in the model archives.

The seven-model comparison artifacts are in the study archive.
The complete archive is
[`efficiency_study_2026_10_20/`](agents/Deep_Q-Network/efficiency_study_2026_10_20/README.md),
with separate evidence folders for every model, a shared source snapshot, and
`best_model.json` identifying the recommended checkpoint. Model 4 (curriculum)
led this benchmark with 60% completion and 93.74 mean standardized reward on
100 matching unseen courses. Original model artifacts are preserved.

The training tools now support `--action-repeat 2`, `--curriculum`,
`--learning-rate 0.0001`, and `--fruit-reward 4`. Evaluation uses the original
fruit reward even when training uses a larger bonus. To repeat the complete
study, choose a new output folder; existing study plans are protected from overwrite:

```bash
python agents/Deep_Q-Network/run_efficiency_study.py --folder agents/Deep_Q-Network/efficiency_study_repeat
python agents/Deep_Q-Network/summarize_efficiency_study.py --folder agents/Deep_Q-Network/efficiency_study_repeat
```

Repeating these commands reuses the documented benchmark seeds. A new unseen
benchmark is needed for an independent confirmation after tuning on these results.

## Custom PyTorch PPO

The implementation in `agents/Proximal_Policy_Optimization/` follows `agents/Proximal_Policy_Optimization/PPO_Platformer_Implementation_Spec.docx`:
128×128 Tanh actor–critic, categorical actions, independent synchronous game
instances, fresh on-policy rollouts, terminal-aware GAE, PPO clipping, optional
old-value clipping, entropy bonus, gradient clipping, KL guard and learning-rate
decay. It uses the existing 60-feature `gap_geometry_v1` observation encoder.
Inputs already lie in [-1, 1]; no additional running normalization is fitted.

Install the project dependencies, then run from the repository root:

```bash
python -m pip install -r requirements.txt
python -m agents.Proximal_Policy_Optimization.train_ppo --config agents/Proximal_Policy_Optimization/config/ppo.json --run-dir agents/Proximal_Policy_Optimization/runs/ppo_seed7 --seed 7
python -m agents.Proximal_Policy_Optimization.evaluate --algo ppo --checkpoint agents/Proximal_Policy_Optimization/runs/ppo_seed7/best.pt --output agents/Proximal_Policy_Optimization/runs/ppo_seed7/test.json
python -m agents.Proximal_Policy_Optimization.plot_training agents/Proximal_Policy_Optimization/runs/ppo_seed7
```

Configuration uses JSON and needs no YAML dependency. `--timesteps` overrides the
transition budget; `--device cpu`, `cuda`, or `mps` selects the PyTorch device when
available. Defaults are four environments, 128 steps per environment, four update
epochs and minibatches of 128. The exact transition budget must be divisible by
the number of environments. The last rollout is shortened to meet that budget.
A transition is one decision, executing up to `action_repeat` physics ticks.
`metrics.csv` records both budgets, throughput, collection/update time, losses,
entropy, approximate KL, clip fraction, gradient norm and learning rate.
`episodes.json` records outcomes, fruit, progress and individual reward components.
`validation.json` records fixed-seed completion and raw gameplay measures.

PPO uses DQN's `CourseTrainingEnv` reward and decision cadence, including the
existing five controls: left, right, jump, double jump, idle. Jump actions retain
horizontal velocity and obey the engine's grounded/double-jump conditions.
Training seeds advance within 0–99,999; validation uses 100,000 onward, with
100,000–100,004 by default; default final tests use 200,000–200,009. Curriculum
is optional and applies only during training. Validation always uses full courses
and the standard fruit reward of 2. No physics or collision changes are needed.

Each run writes `latest.pt`, periodic `checkpoint_<steps>.pt`, `best.pt` ranked by
validation completion then progress, and `final.pt`. Checkpoints contain policy,
optimizer, configuration, counters and PyTorch RNG state. Resume into a **fresh**
run directory:

```bash
python -m agents.Proximal_Policy_Optimization.train_ppo --resume agents/Proximal_Policy_Optimization/runs/ppo_seed7/latest.pt --run-dir agents/Proximal_Policy_Optimization/runs/ppo_seed7_continued --timesteps 500000
```

Resume adds the requested decision budget, retaining policy/optimizer/counters;
it restarts simulator episodes, course scheduling and the learning-rate schedule.
It is not a bit-for-bit continuation of the interrupted trajectories. Configuration
must otherwise match. Existing run metrics are protected from overwrite.

For visible playback, add `--render` to the evaluation command. For a quick
verification, create a config overriding `total_timesteps` to 4096,
`max_episode_steps` to 512 and `validation_episodes` to 2. This checks execution,
not learned completion skill. Run `python -m pytest -q` for correctness checks.

For a controlled comparison, retrain each algorithm from scratch with at least
three independent training seeds and equal transition budgets, action repeat,
reward weights, curriculum, horizon and course distribution. The common evaluator
supports `--algo dqn`, `random` and `right` as well as PPO; use `--seed-list` with
a JSON array to select identical held-out courses. Use current 60-input DQN
checkpoints. Pass matching `--max-episode-steps` and action repeat when needed.

```bash
python -m agents.Proximal_Policy_Optimization.evaluate --algo dqn --checkpoint path/to/dqn.zip --output runs/dqn_test.json
python -m agents.Proximal_Policy_Optimization.compare agents/Proximal_Policy_Optimization/runs/ppo_seed7/test.json runs/dqn_test.json --output runs/comparison.json
```

The comparison tool rejects mismatched evaluation protocols and reports means and
sample standard deviations across reports (use one report per training run).
Training budgets are listed so unequal budgets remain visible. A single run has
no estimated standard deviation. Repeat experiments at equal wall time separately;
this implementation does not claim a PPO advantage without those experiments.
Use configuration copies for reward, entropy, network-size, vectorization and
curriculum ablations. Pixel policies, subprocess vectorization, checkpoint-start
curricula, automated hyperparameter search and a completed multi-seed research
study remain follow-up experiments.

## PPO efficiency study

Run the controlled study from the project root:

```bash
python -m agents.Proximal_Policy_Optimization.run_efficiency_study
```

Artifacts are written under
`agents/Proximal_Policy_Optimization/efficiency_study_2026_11_09/`.
The runner preserves the completed `runs/ppo_seed7` baseline, trains five isolated
ablations and a validation-selected combination, repeats the baseline and the
validation winner with three training seeds, evaluates on matched DQN courses
and fresh confirmation courses, and benchmarks 1/4/8 synchronous environments.
Training runs execute sequentially. This is a sustained experiment, not a short
smoke test. Per-run `training_console.log` and study `status.json` show progress.

Read `plan.json` for predeclared budgets and selection rules. The final `report.md`,
`results.csv`, `seed_aggregates.json`, and plots contain the findings. Re-running
the command skips completed runs; incomplete runs are preserved and reported
rather than overwritten. Use `--folder` to start a separate study.

Completed PPO study: see
`agents/Proximal_Policy_Optimization/efficiency_study_2026_11_09/findings.md`
and `report.md`. None of the five isolated changes beat the baseline on validation
or on the 100 matched test courses. The unchanged combination stage reproduced
the original policy exactly; defaults remain unchanged. Three-seed confirmation
showed substantial variation. Full checkpoints, plots, seed lists and provenance
are retained in the study directory.

## PPO exploration tuning

Run the entropy-only study with:

```bash
python -m agents.Proximal_Policy_Optimization.run_exploration_study
```

It compares the existing constant `entropy_coef=0.01` baseline with no entropy
bonus, a constant `0.03` bonus, and `0.05` held for 20% of training then linearly
decayed to `0.001`. Other learning settings and the 500,000-decision budget stay
fixed. Existing baseline checkpoints are reselected at the matching 50,000-step
cadence on 25 new validation seeds (110000–110024); final tests use 100 untouched
seeds (1000000–1000099). A new validation winner receives two additional training
seeds. Original training runs and earlier reports are preserved.

Optional training configuration fields:

- `entropy_schedule`: `constant` (default) or `linear`.
- `entropy_final_coef`: nonnegative final bonus for a linear schedule.
- `entropy_hold_fraction`: fraction of training before decay begins, default 0.
- `validation_seed_start`: first validation course seed, default 100000.

`metrics.csv` now includes the actual entropy coefficient and the fraction of
sampled actions for each of the five engine controls. Checkpoints retain the
initial schedule configuration. The study's `report.md`, `results.csv`,
`seed_aggregates.json` and `exploration_curves.png` are stored under
`agents/Proximal_Policy_Optimization/exploration_study_2026_11_09/`.

The completed exploration study found no improvement: on the 100 new final
courses, seed-7 completion was 14% for the existing 0.01 bonus, 2% without a
bonus, 0% with constant 0.03, and 2% with hold-then-decay. The default remains
0.01. New candidates were screened with one training seed; the baseline's
three-seed results remain highly variable. See the study report for limits.
