# model_5_lower_learning_rate: Lower learning rate

Method: Gap features + learning rate 0.0001 instead of 0.0003.

Measured on the same 100 full-difficulty courses, seeds 400000–400099: **5% completion**, **-24.03 mean standardized reward**, **5.10 mean fruit**.

Strengths:

- Collected 5.10 fruit per episode versus Model 2's 4.16.
- Fewer falls than Model 2 (5 versus 12).

Weaknesses:

- Completed 28 fewer courses than Model 2.
- Mean standardized reward was 15.75 lower than Model 2.
- Higher mean damage than Model 2 (2.58 versus 1.45).
- Failed to finish 95 of 100 courses: 5 falls, 72 defeats, 18 timeouts.
- Fruit collection (5.10) was below the best model's 5.59.

The selected checkpoint has 500,000 decisions; action repeat is 4. Models 1 and 2 reuse preserved historical weights; Models 3–7 are fresh runs with training seed 7.

Files: `saved_models/` (best/final weights and configuration), `tensorboard_logs/`, `training_results.png`, `evaluation_results.json` (all outcomes), `method.json`, and `walkthrough/` (GIF and full trajectory). Console logs are also preserved. The shared immutable source snapshot and experiment plan are in the parent folder.

The recording shows the first completed benchmark course, or the highest-progress failure if there are no completions. It plays at 3× speed. The full evaluation includes failed courses.

Playback from the repository root:

```bash
.venv/bin/python agents/Deep_Q-Network/watch_dqn.py --model agents/Deep_Q-Network/efficiency_study_2026_10_20/model_5_lower_learning_rate/saved_models/dqn_platformer_best.zip
```
