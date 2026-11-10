"""Verify experimental selection does not use held-out results and budgets match."""
from agents.Proximal_Policy_Optimization.run_efficiency_study import config_for, choose_combination


def validation(completion, progress):
    return {'summary': dict(completion_rate=completion, mean_progress_pixels=progress)}


def test_combination_uses_only_strict_validation_improvements():
    baseline = validation(.2, 3301)
    candidates = [({'id':'rate', 'changes':{'learning_rate':.0001}}, validation(.4, 2000)),
                  ({'id':'gamma', 'changes':{'gamma':.995}}, validation(.2, 3400)),
                  ({'id':'repeat', 'changes':{'action_repeat':2}}, validation(0, 4000)),
                  ({'id':'network', 'changes':{'hidden_size':64}}, validation(.2, 3301))]
    result = choose_combination(baseline, candidates)
    assert result['accepted'] == ['rate', 'gamma']
    assert result['changes'] == {'learning_rate':.0001, 'gamma':.995}


def test_budget_and_validation_cadence_use_physics_ticks():
    base = dict(action_repeat=4, num_envs=4)
    for repeat, decisions in [(4,500000), (2,1000000)]:
        config = config_for(base, {'action_repeat':repeat}, 17, 2000000)
        assert config['total_timesteps'] == decisions
        assert config['total_timesteps']*repeat == 2000000
        assert config['validation_freq']*repeat == 100000
        assert config['seed'] == 17
