"""Measure 1/4/8-environment throughput using equal-sized PPO update batches."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from agents.Proximal_Policy_Optimization.envs import make_env
from agents.Proximal_Policy_Optimization.evaluate import load_policy
from agents.Proximal_Policy_Optimization.ppo.trainer import PPOTrainer


def benchmark(checkpoint, output, decisions=8192, trials=3):
    torch.set_num_threads(1)
    model, base, _ = load_policy(checkpoint)
    results = []
    for trial in range(trials):
        # Rotate order to reduce systematic thermal/order effects.
        counts = [1, 4, 8]
        counts = counts[trial:] + counts[:trial]
        for count in counts:
            config = dict(base, num_envs=count, rollout_steps=512//count, curriculum=False)
            envs = []
            try:
                for i in range(count):
                    envs.append(make_env(config, i))
                trainer = PPOTrainer(envs, config)
                trainer.model.load_state_dict(model.state_dict())
                for _ in range(4):
                    trainer.update(trainer.collect_rollout(config['rollout_steps']))
                collect_time = update_time = 0.
                initial_ticks = trainer.physics_ticks
                started = time.perf_counter()
                for _ in range(decisions//512):
                    tick = time.perf_counter()
                    rollout = trainer.collect_rollout(config['rollout_steps'])
                    collect_time += time.perf_counter()-tick
                    tick = time.perf_counter()
                    trainer.update(rollout)
                    update_time += time.perf_counter()-tick
                elapsed = time.perf_counter()-started
                row = dict(num_envs=count, trial=trial, decisions=decisions,
                           actual_physics_ticks=trainer.physics_ticks-initial_ticks,
                           collection_seconds=collect_time, update_seconds=update_time,
                           elapsed_seconds=elapsed, decisions_per_second=decisions/elapsed)
                results.append(row)
                print(json.dumps(row), flush=True)
            finally:
                for env in envs:
                    env.close()
    summary = {str(n): dict(mean_decisions_per_second=float(np.mean([
        r['decisions_per_second'] for r in results if r['num_envs']==n])),
        std_decisions_per_second=float(np.std([
            r['decisions_per_second'] for r in results if r['num_envs']==n], ddof=1)))
               for n in (1, 4, 8)}
    Path(output).write_text(json.dumps(dict(checkpoint=str(checkpoint), batch_size=512,
        warmup_decisions=2048, trials=results, summary=summary,
        scope='CPU collection plus optimization only; fixed full difficulty; no checkpoint or evaluation I/O; does not establish policy skill'), indent=2)+'\n')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    benchmark(args.checkpoint, args.output)
