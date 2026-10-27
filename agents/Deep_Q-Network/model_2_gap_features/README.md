# Model 2: gap-feature DQN

Recorded October 20, 2026. This is a separate archive of the second model. The first model remains in the parent directory's `saved_models/`, `tensorboard_logs/`, and `walkthrough/`, with its original evaluation and chart.

| Metric on the same 100 unseen courses | First model | Model 2 |
| --- | --- | --- |
| Completion | 13 / 100 | 26 / 100 |
| Mean shaped reward | -46.63 | -20.38 |
| Mean fruit | 4.49 | 3.79 |
| Falls | 21 | 12 |

Completion doubled in this experiment; fruit collection decreased. These are results from one new training run, not proof of consistent gains across repeated runs.

## Saved files

- `saved_models/dqn_platformer_best.zip`: validation-selected 60-input policy, at 50,000 decisions.
- `saved_models/dqn_platformer_final.zip`: final policy after 500,000 training decisions; the reported test results use the selected policy.
- `saved_models/config.json` and `validation.json`: original training configuration and checkpoint selection history.
- `tensorboard_logs/episodes.monitor.csv`: full episode training log.
- `training_results.png`: training curve.
- `evaluation_results.json`: all 100 test outcomes, seeds 300000–300099.
- `baseline_evaluation_100_courses.json`: original policy evaluated on those same 100 courses.
- `baseline_recheck_20_courses.json`: reproduction of the original 20-course results.
- `results.md` and `evidence/`: full comparison report, chart, and detailed comparison data.
- `walkthrough/walkthrough.gif` and `walkthrough.json`: actual model-controlled gameplay and full trajectory on seed 300001. This is the first successful course in the evaluation list, chosen to illustrate a completion; the 26% success rate includes failures.
- `source_snapshot/`: code reference for the 60-input environment and training tools. Shared game assets remain in the main repository; this snapshot is not a standalone installation.
- `original_model_integrity.json`: SHA-256 hashes confirming the original model's artifacts were preserved.
- `archive_manifest.json`: checksums and metadata for this archive.

## Playback

Run from the repository root:

```bash
.venv/bin/python agents/Deep_Q-Network/watch_dqn.py --model agents/Deep_Q-Network/model_2_gap_features/saved_models/dqn_platformer_best.zip --seed 300001
```

The saved GIF can be viewed directly without launching the game. The original 51-input model is preserved and can be evaluated with `evaluate_dqn.py --legacy-observations`; the current playback script expects the newer 60-input format.

## Reproduce the evaluation

Use a fresh output filename to keep this archive's evaluation intact:

```bash
.venv/bin/python agents/Deep_Q-Network/evaluate_dqn.py --model agents/Deep_Q-Network/model_2_gap_features/saved_models/dqn_platformer_best.zip --episodes 100 --seed 300000 --output /tmp/model_2_evaluation_recheck.json
```

The copied evaluation files retain their original experiment model paths as provenance. The archived weights are byte-for-byte copies of those experiment weights.
