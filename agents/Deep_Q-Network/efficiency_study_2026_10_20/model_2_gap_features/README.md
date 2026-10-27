# model_2_gap_features: Gap features

Method: Exact gap and landing observations (60 inputs).

Measured on the same 100 full-difficulty courses, seeds 400000–400099: **33% completion**, **-8.28 mean standardized reward**, **4.16 mean fruit**.

Strengths:

- Completed 20 more courses than the original model and earned 48.12 higher mean reward.

Weaknesses:

- Failed to finish 67 of 100 courses: 12 falls, 22 defeats, 33 timeouts.
- Fruit collection (4.16) was below the best model's 5.59.

The selected checkpoint has 50,000 decisions; action repeat is 4. Models 1 and 2 reuse preserved historical weights; Models 3–7 are fresh runs with training seed 7.

Files: `saved_models/` (best/final weights and configuration), `tensorboard_logs/`, `training_results.png`, `evaluation_results.json` (all outcomes), `method.json`, and `walkthrough/` (GIF and full trajectory). Console logs are also preserved. The shared immutable source snapshot and experiment plan are in the parent folder.

The recording shows the first completed benchmark course, or the highest-progress failure if there are no completions. It plays at 3× speed. The full evaluation includes failed courses.

Playback from the repository root:

```bash
.venv/bin/python agents/Deep_Q-Network/watch_dqn.py --model agents/Deep_Q-Network/efficiency_study_2026_10_20/model_2_gap_features/saved_models/dqn_platformer_best.zip
```
