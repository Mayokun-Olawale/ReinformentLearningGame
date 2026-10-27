# model_4_curriculum: Curriculum

Method: Gap features + gradually harder training courses.

Measured on the same 100 full-difficulty courses, seeds 400000–400099: **60% completion**, **93.74 mean standardized reward**, **4.99 mean fruit**.

Strengths:

- Highest completion rate in this benchmark (reward breaks completion ties).
- Highest mean standardized reward in this benchmark.
- Collected 4.99 fruit per episode versus Model 2's 4.16.
- Completed 27 more courses than Model 2 on the same 100 layouts.
- Mean standardized reward was 102.02 higher than Model 2.
- No timeouts in the 100-course test.
- The selected checkpoint was learned during difficulty 0.2 training and was evaluated at full difficulty.

Weaknesses:

- Higher mean damage than Model 2 (1.52 versus 1.45).
- Failed to finish 40 of 100 courses: 22 falls, 18 defeats, 0 timeouts.
- Fruit collection (4.99) was below the best model's 5.59.
- The curriculum changes the training layout distribution; improved full-difficulty generalization is not guaranteed.
- Later full-difficulty training did not replace this earlier checkpoint under the fixed validation selection rule.

The selected checkpoint has 100,000 decisions; action repeat is 4. Models 1 and 2 reuse preserved historical weights; Models 3–7 are fresh runs with training seed 7.

Files: `saved_models/` (best/final weights and configuration), `tensorboard_logs/`, `training_results.png`, `evaluation_results.json` (all outcomes), `method.json`, and `walkthrough/` (GIF and full trajectory). Console logs are also preserved. The shared immutable source snapshot and experiment plan are in the parent folder.

The recording shows the first completed benchmark course, or the highest-progress failure if there are no completions. It plays at 3× speed. The full evaluation includes failed courses.

Playback from the repository root:

```bash
.venv/bin/python agents/Deep_Q-Network/watch_dqn.py --model agents/Deep_Q-Network/efficiency_study_2026_10_20/model_4_curriculum/saved_models/dqn_platformer_best.zip
```
