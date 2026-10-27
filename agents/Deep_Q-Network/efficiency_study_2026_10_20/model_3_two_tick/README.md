# model_3_two_tick: Two-tick decisions

Method: Gap features + decisions every two physics ticks.

Measured on the same 100 full-difficulty courses, seeds 400000–400099: **19% completion**, **43.02 mean standardized reward**, **5.59 mean fruit**.

Strengths:

- Highest mean fruit collection in this benchmark.
- Collected 5.59 fruit per episode versus Model 2's 4.16.
- Mean standardized reward was 51.30 higher than Model 2.
- Fewer falls than Model 2 (3 versus 12).

Weaknesses:

- Completed 14 fewer courses than Model 2.
- Higher mean damage than Model 2 (2.71 versus 1.45).
- Failed to finish 81 of 100 courses: 3 falls, 76 defeats, 2 timeouts.
- Twice as many decisions and potential optimizer updates for the same nominal physics budget; temporal discounting is still per decision.

The selected checkpoint has 850,000 decisions; action repeat is 2. Models 1 and 2 reuse preserved historical weights; Models 3–7 are fresh runs with training seed 7.

Files: `saved_models/` (best/final weights and configuration), `tensorboard_logs/`, `training_results.png`, `evaluation_results.json` (all outcomes), `method.json`, and `walkthrough/` (GIF and full trajectory). Console logs are also preserved. The shared immutable source snapshot and experiment plan are in the parent folder.

The recording shows the first completed benchmark course, or the highest-progress failure if there are no completions. It plays at 3× speed. The full evaluation includes failed courses.

Playback from the repository root:

```bash
.venv/bin/python agents/Deep_Q-Network/watch_dqn.py --model agents/Deep_Q-Network/efficiency_study_2026_10_20/model_3_two_tick/saved_models/dqn_platformer_best.zip
```
