# model_7_combined: Combined methods

Method: Gap features + two-tick decisions + curriculum + lower learning rate + fruit bonus 4.

Measured on the same 100 full-difficulty courses, seeds 400000–400099: **17% completion**, **26.99 mean standardized reward**, **3.33 mean fruit**.

Strengths:

- Mean standardized reward was 35.28 higher than Model 2.
- The selected checkpoint was learned during difficulty 0.2 training and was evaluated at full difficulty.

Weaknesses:

- Completed 16 fewer courses than Model 2.
- Higher mean damage than Model 2 (1.69 versus 1.45).
- Failed to finish 83 of 100 courses: 53 falls, 26 defeats, 4 timeouts.
- Fruit collection (3.33) was below the best model's 5.59.
- The curriculum changes the training layout distribution; improved full-difficulty generalization is not guaranteed.
- Later full-difficulty training did not replace this earlier checkpoint under the fixed validation selection rule.
- Twice as many decisions and potential optimizer updates for the same nominal physics budget; temporal discounting is still per decision.
- Its training reward curve uses a larger fruit bonus and cannot be compared directly with other training reward curves. Test rewards are standardized.

The selected checkpoint has 50,000 decisions; action repeat is 2. Models 1 and 2 reuse preserved historical weights; Models 3–7 are fresh runs with training seed 7.

Files: `saved_models/` (best/final weights and configuration), `tensorboard_logs/`, `training_results.png`, `evaluation_results.json` (all outcomes), `method.json`, and `walkthrough/` (GIF and full trajectory). Console logs are also preserved. The shared immutable source snapshot and experiment plan are in the parent folder.

The recording shows the first completed benchmark course, or the highest-progress failure if there are no completions. It plays at 3× speed. The full evaluation includes failed courses.

Playback from the repository root:

```bash
.venv/bin/python agents/Deep_Q-Network/watch_dqn.py --model agents/Deep_Q-Network/efficiency_study_2026_10_20/model_7_combined/saved_models/dqn_platformer_best.zip
```
