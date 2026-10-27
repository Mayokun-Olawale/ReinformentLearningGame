# model_1_baseline: Original DQN

Method: Original 51 inputs.

Measured on the same 100 full-difficulty courses, seeds 400000–400099: **13% completion**, **-56.40 mean standardized reward**, **4.11 mean fruit**.

Strengths:

- Lower mean damage than Model 2 (1.31 versus 1.45).

Weaknesses:

- Completed 20 fewer courses than Model 2.
- Mean standardized reward was 48.12 lower than Model 2.
- Failed to finish 87 of 100 courses: 28 falls, 21 defeats, 38 timeouts.
- Fruit collection (4.11) was below the best model's 5.59.

The selected checkpoint has 125,000 decisions; action repeat is 4. Models 1 and 2 reuse preserved historical weights; Models 3–7 are fresh runs with training seed 7.

Files: `saved_models/` (best/final weights and configuration), `tensorboard_logs/`, `training_results.png`, `evaluation_results.json` (all outcomes), `method.json`, and `walkthrough/` (GIF and full trajectory). Console logs are also preserved. The shared immutable source snapshot and experiment plan are in the parent folder.

The recording shows the first completed benchmark course, or the highest-progress failure if there are no completions. It plays at 3× speed. The full evaluation includes failed courses.

Playback from the repository root:

```bash
.venv/bin/python agents/Deep_Q-Network/watch_dqn.py --model agents/Deep_Q-Network/efficiency_study_2026_10_20/model_1_baseline/saved_models/dqn_platformer_best.zip --legacy-observations
```
