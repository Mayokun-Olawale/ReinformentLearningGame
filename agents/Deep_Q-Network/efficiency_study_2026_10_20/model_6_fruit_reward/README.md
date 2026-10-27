# model_6_fruit_reward: Larger fruit bonus

Method: Gap features + training fruit bonus 4 instead of 2.

Measured on the same 100 full-difficulty courses, seeds 400000–400099: **1% completion**, **-48.31 mean standardized reward**, **4.54 mean fruit**.

Strengths:

- Collected 4.54 fruit per episode versus Model 2's 4.16.

Weaknesses:

- Completed 32 fewer courses than Model 2.
- Mean standardized reward was 40.03 lower than Model 2.
- Higher mean damage than Model 2 (1.90 versus 1.45).
- Failed to finish 99 of 100 courses: 28 falls, 44 defeats, 27 timeouts.
- Fruit collection (4.54) was below the best model's 5.59.
- Its training reward curve uses a larger fruit bonus and cannot be compared directly with other training reward curves. Test rewards are standardized.

The selected checkpoint has 75,000 decisions; action repeat is 4. Models 1 and 2 reuse preserved historical weights; Models 3–7 are fresh runs with training seed 7.

Files: `saved_models/` (best/final weights and configuration), `tensorboard_logs/`, `training_results.png`, `evaluation_results.json` (all outcomes), `method.json`, and `walkthrough/` (GIF and full trajectory). Console logs are also preserved. The shared immutable source snapshot and experiment plan are in the parent folder.

The recording shows the first completed benchmark course, or the highest-progress failure if there are no completions. It plays at 3× speed. The full evaluation includes failed courses.

Playback from the repository root:

```bash
.venv/bin/python agents/Deep_Q-Network/watch_dqn.py --model agents/Deep_Q-Network/efficiency_study_2026_10_20/model_6_fruit_reward/saved_models/dqn_platformer_best.zip
```
