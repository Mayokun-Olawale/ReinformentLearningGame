"""Exploration schedules and safety of configuration boundaries."""
import json
import pytest
from agents.Proximal_Policy_Optimization.train_ppo import entropy_coefficient, validate, DEFAULT_CONFIG


def test_legacy_constant_and_zero_entropy():
    for value in (0., .01, .03):
        for progress in (0, .5, 1):
            assert entropy_coefficient({'entropy_coef':value}, progress) == value


def test_hold_then_linear_entropy_schedule():
    config = dict(entropy_coef=.05, entropy_schedule='linear', entropy_final_coef=.001,
                  entropy_hold_fraction=.2)
    assert entropy_coefficient(config, 0) == .05
    assert entropy_coefficient(config, .2) == .05
    assert entropy_coefficient(config, .6) == pytest.approx(.0255)
    assert entropy_coefficient(config, 1) == pytest.approx(.001)
    assert entropy_coefficient(config, 2) == pytest.approx(.001)


@pytest.mark.parametrize('changes', [dict(entropy_schedule='unknown'),
    dict(entropy_hold_fraction=1),dict(entropy_schedule='linear',entropy_final_coef=-1),
    dict(validation_seed_start=99999)])
def test_invalid_schedule_or_validation_seed(changes):
    config = json.loads(DEFAULT_CONFIG.read_text())
    with pytest.raises(ValueError):
        validate(dict(config, **changes))


def test_scheduled_training_logs_bonus_without_changing_saved_start(tmp_path):
    import csv
    from agents.Proximal_Policy_Optimization.train_ppo import train
    from agents.Proximal_Policy_Optimization.evaluate import load_policy
    path=train(dict(total_timesteps=32,num_envs=2,rollout_steps=8,minibatch_size=8,
                    max_episode_steps=24,validation_freq=16,validation_episodes=1,
                    validation_seed_start=110000,entropy_coef=.05,entropy_schedule='linear',
                    entropy_hold_fraction=.2,entropy_final_coef=.001),tmp_path/'scheduled')
    with (path.parent/'metrics.csv').open() as stream:
        rows=list(csv.DictReader(stream))
    assert [float(r['entropy_coef']) for r in rows]==pytest.approx([.05,.031625])
    assert sum(float(rows[0][f'action_fraction_{i}']) for i in range(5))==pytest.approx(1.)
    assert load_policy(path)[1]['entropy_coef']==.05
    summary=json.loads((path.parent/'summary.json').read_text())
    assert summary['validation_seeds']==[110000]


def test_historical_checkpoints_match_new_validation_cadence(tmp_path):
    from agents.Proximal_Policy_Optimization.run_exploration_study import checkpoint_subset
    for step in [i*25088 for i in range(1,20)]+[500000]:
        (tmp_path/f'checkpoint_{step}.pt').touch()
    chosen=checkpoint_subset(tmp_path)
    assert [int(p.stem.split('_')[-1]) for p in chosen]==[50176*i for i in range(1,10)]+[500000]
