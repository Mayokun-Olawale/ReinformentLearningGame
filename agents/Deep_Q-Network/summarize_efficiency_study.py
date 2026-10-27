"""Build presentation evidence and a report from the study's saved evaluations."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from training_paths import PROJECT_ROOT, TRAINING_DIR


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(folder):
    folder = Path(folder).resolve()
    plan = json.loads((folder/'plan.json').read_text())
    records = []
    for spec in plan['variants']:
        dest = folder/spec['id']
        evaluation = json.loads((dest/'evaluation_results.json').read_text())
        config = json.loads((dest/'saved_models/config.json').read_text())
        assert [r['seed'] for r in evaluation['episodes']] == list(range(400000,400100))
        assert evaluation['evaluation_fruit_reward'] == 2
        assert evaluation['action_repeat'] == spec['repeat']
        successes = [row for row in evaluation['episodes'] if row['outcome']=='completed']
        stats = dict(evaluation['summary'], completions=len(successes),
                     defeats=sum(r['outcome']=='defeated' for r in evaluation['episodes']),
                     completed_mean_ticks=float(np.mean([r['ticks'] for r in successes])) if successes else None,
                     completed_mean_fruit=float(np.mean([r['fruit'] for r in successes])) if successes else None)
        records.append(dict(spec=spec, evaluation=evaluation, config=config, stats=stats))
    names = [r['spec']['name'] for r in records]
    best_completion = max(records,key=lambda r:(r['stats']['completion_rate'],r['stats']['mean_reward']))
    best_reward = max(records,key=lambda r:(r['stats']['mean_reward'],r['stats']['completion_rate']))
    best_fruit = max(records,key=lambda r:r['stats']['mean_fruit'])
    pareto = [r['spec']['id'] for r in records if not any(
        q['stats']['completion_rate'] >= r['stats']['completion_rate'] and q['stats']['mean_reward'] >= r['stats']['mean_reward']
        and (q['stats']['completion_rate'] > r['stats']['completion_rate'] or q['stats']['mean_reward'] > r['stats']['mean_reward'])
        for q in records)]
    indices = np.random.default_rng(7).integers(0,100,size=(10000,100))
    for record in records:
        record['paired_comparisons'] = {}
        for ref in records[:2]:
            intervals = {}
            for metric, getter in [('completion_rate',lambda r:float(r['outcome']=='completed')),
                                   ('mean_reward',lambda r:r['reward']),('mean_fruit',lambda r:r['fruit'])]:
                delta = np.array([getter(row) for row in record['evaluation']['episodes']]) - np.array([getter(row) for row in ref['evaluation']['episodes']])
                intervals[metric] = dict(change=float(delta.mean()),bootstrap_95_percent=np.quantile(delta[indices].mean(axis=1),[0.025,0.975]).tolist())
            record['paired_comparisons'][ref['spec']['id']] = intervals
    # Validate recordings against their measured course, and preservation against pre-study hashes.
    for record in records:
        dest=folder/record['spec']['id']
        recording=json.loads((dest/'walkthrough/walkthrough.json').read_text())
        example=next(r for r in record['evaluation']['episodes'] if r['seed']==recording['seed'])
        assert all(recording[key]==example[key] for key in ('outcome','ticks','fruit','damage','progress_pixels'))
        assert len(recording['trajectory'])==recording['ticks']
    preserved=json.loads((folder/'preserved_originals.json').read_text())
    assert all(sha(PROJECT_ROOT/name)==digest for name,digest in preserved.items()), 'An original artifact changed'
    evidence=folder/'evidence'
    evidence.mkdir(exist_ok=True)
    summary=dict(plan=plan,models=records,best_completion=best_completion['spec']['id'],
                 best_reward=best_reward['spec']['id'],best_fruit=best_fruit['spec']['id'],
                 pareto_front_completion_and_reward=pareto,original_artifact_files_verified=len(preserved))
    winner_path = folder/best_completion['spec']['id']/'saved_models/dqn_platformer_best.zip'
    recommendation = dict(model=best_completion['spec']['id'],
                          checkpoint=str(winner_path.relative_to(PROJECT_ROOT)),
                          checkpoint_sha256=sha(winner_path),
                          also_best_mean_reward=best_completion is best_reward,
                          metrics=best_completion['stats'],
                          scope='Observed best on this 100-course benchmark; one training seed per method')
    (folder/'best_model.json').write_text(json.dumps(recommendation,indent=2)+'\n')
    (evidence/'all_model_results.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (evidence/'model_summary.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['model','method','completion_rate','completions','mean_reward','mean_fruit','mean_damage','falls','defeats','timeouts','mean_progress_pixels','selected_decisions','action_repeat'])
        writer.writeheader()
        for r in records:
            writer.writerow(dict(model=r['spec']['id'],method=r['spec']['method'],
                **{k:r['stats'][k] for k in ('completion_rate','completions','mean_reward','mean_fruit','mean_damage','falls','defeats','timeouts','mean_progress_pixels')},
                selected_decisions=r['evaluation']['model_decisions'],action_repeat=r['spec']['repeat']))
    with (evidence/'course_by_course.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['model','seed','outcome','reward','ticks','fruit','damage','progress_pixels'])
        writer.writeheader()
        for r in records:
            for row in r['evaluation']['episodes']:
                writer.writerow(dict(model=r['spec']['id'],**row))
    colors=['#6f7c8a','#33849b','#5a9c74','#af8b36','#8968ab','#c27653','#245c46']
    for filename,metrics in [('completion_and_reward.png',[('completion_rate','Completed courses (%)',100),('mean_reward','Mean standardized reward',1)]),
                             ('fruit_and_damage.png',[('mean_fruit','Mean fruit per course',1),('mean_damage','Mean damage per course',1)])]:
        fig,axes=plt.subplots(1,2,figsize=(14,6))
        for ax,(key,title,scale) in zip(axes,metrics):
            bars=ax.barh(names,[r['stats'][key]*scale for r in records],color=colors)
            ax.bar_label(bars,fmt='%.2f',padding=4)
            ax.invert_yaxis()
            ax.axvline(0,color='black',linewidth=0.6)
            ax.margins(x=0.22)
            ax.set_title(title)
            ax.grid(axis='x',alpha=0.2)
            ax.set_axisbelow(True)
        fig.suptitle('Seven DQN policies: identical 100 full-difficulty test courses (400000–400099)')
        fig.tight_layout()
        fig.savefig(evidence/filename,dpi=180)
        plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,6))
    left=np.zeros(len(records))
    for key,label,color in [('completions','Completed','#329664'),('falls','Fell','#d68b36'),('defeats','Defeated','#bb5353'),('timeouts','Timed out','#8a8f99')]:
        values=np.array([r['stats'][key] for r in records])
        bars=ax.barh(names,values,left=left,label=label,color=color)
        ax.bar_label(bars,labels=[str(v) if v else '' for v in values],label_type='center',fontsize=9)
        left+=values
    ax.invert_yaxis()
    ax.set_xlabel('Number of test courses')
    ax.set_title('Every test outcome, including failures')
    ax.legend(loc='upper center',bbox_to_anchor=(0.5,-0.1),ncol=4)
    fig.tight_layout()
    fig.savefig(evidence/'all_outcomes.png',dpi=180)
    plt.close(fig)

    gap=records[1]['stats']
    detail=[]
    for record in records:
        spec,stats=record['spec'],record['stats']
        change=stats['completions']-gap['completions']
        reward_change=stats['mean_reward']-gap['mean_reward']
        strength=[]
        weakness=[]
        if record is best_completion:
            strength.append('Highest completion rate in this benchmark (reward breaks completion ties).')
        if record is best_reward:
            strength.append('Highest mean standardized reward in this benchmark.')
        if record is best_fruit:
            strength.append('Highest mean fruit collection in this benchmark.')
        if record is records[1]:
            baseline = records[0]['stats']
            strength.append(f"Completed {stats['completions']-baseline['completions']} more courses than the original model and earned {stats['mean_reward']-baseline['mean_reward']:.2f} higher mean reward.")
        if stats['mean_fruit'] > gap['mean_fruit'] + 0.005:
            strength.append(f"Collected {stats['mean_fruit']:.2f} fruit per episode versus Model 2's {gap['mean_fruit']:.2f}.")
        if change>0:
            strength.append(f"Completed {change} more courses than Model 2 on the same 100 layouts.")
        elif change<0:
            weakness.append(f"Completed {-change} fewer courses than Model 2.")
        if reward_change>0.005:
            strength.append(f'Mean standardized reward was {reward_change:.2f} higher than Model 2.')
        elif reward_change < -0.005:
            weakness.append(f'Mean standardized reward was {-reward_change:.2f} lower than Model 2.')
        if stats['falls']<gap['falls']:
            strength.append(f"Fewer falls than Model 2 ({stats['falls']} versus {gap['falls']}).")
        if stats['mean_damage'] > gap['mean_damage'] + 0.05:
            weakness.append(f"Higher mean damage than Model 2 ({stats['mean_damage']:.2f} versus {gap['mean_damage']:.2f}).")
        elif stats['mean_damage'] < gap['mean_damage'] - 0.05:
            strength.append(f"Lower mean damage than Model 2 ({stats['mean_damage']:.2f} versus {gap['mean_damage']:.2f}).")
        if stats['timeouts'] == 0:
            strength.append('No timeouts in the 100-course test.')
        if not strength:
            strength.append(f"Completed {stats['completions']} courses with {stats['mean_progress_pixels']:.0f} mean forward pixels across the 100 unseen courses.")
        weakness.append(f"Failed to finish {100-stats['completions']} of 100 courses: {stats['falls']} falls, {stats['defeats']} defeats, {stats['timeouts']} timeouts.")
        if stats['mean_fruit']<max(r['stats']['mean_fruit'] for r in records)-0.005:
            weakness.append(f"Fruit collection ({stats['mean_fruit']:.2f}) was below the best model's {best_fruit['stats']['mean_fruit']:.2f}.")
        if spec.get('curriculum'):
            weakness.append('The curriculum changes the training layout distribution; improved full-difficulty generalization is not guaranteed.')
            stages = record['config'].get('curriculum_stages', [])
            selected_stage = next((stage for stage in reversed(stages)
                                   if stage['decisions'] <= record['evaluation']['model_decisions']), None)
            if selected_stage and selected_stage['difficulty'] < 1:
                strength.append(f"The selected checkpoint was learned during difficulty {selected_stage['difficulty']:.1f} training and was evaluated at full difficulty.")
                weakness.append('Later full-difficulty training did not replace this earlier checkpoint under the fixed validation selection rule.')
        if spec['repeat']==2:
            weakness.append('Twice as many decisions and potential optimizer updates for the same nominal physics budget; temporal discounting is still per decision.')
        if spec['fruit']==4:
            weakness.append('Its training reward curve uses a larger fruit bonus and cannot be compared directly with other training reward curves. Test rewards are standardized.')
        record['strengths']=strength
        record['weaknesses']=weakness
        dest=folder/spec['id']
        legacy=' --legacy-observations' if spec.get('legacy') else ''
        playback=f".venv/bin/python agents/Deep_Q-Network/watch_dqn.py --model {dest.relative_to(PROJECT_ROOT)}/saved_models/dqn_platformer_best.zip{legacy}"
        text=f"# {spec['id']}: {spec['name']}\n\nMethod: {spec['method']}.\n\nMeasured on the same 100 full-difficulty courses, seeds 400000–400099: **{stats['completions']}% completion**, **{stats['mean_reward']:.2f} mean standardized reward**, **{stats['mean_fruit']:.2f} mean fruit**.\n\nStrengths:\n\n"+'\n'.join('- '+s for s in strength)+"\n\nWeaknesses:\n\n"+'\n'.join('- '+s for s in weakness)+f"\n\nThe selected checkpoint has {record['evaluation']['model_decisions']:,} decisions; action repeat is {spec['repeat']}. Models 1 and 2 reuse preserved historical weights; Models 3–7 are fresh runs with training seed 7.\n\nFiles: `saved_models/` (best/final weights and configuration), `tensorboard_logs/`, `training_results.png`, `evaluation_results.json` (all outcomes), `method.json`, and `walkthrough/` (GIF and full trajectory). Console logs are also preserved. The shared immutable source snapshot and experiment plan are in the parent folder.\n\nThe recording shows the first completed benchmark course, or the highest-progress failure if there are no completions. It plays at 3× speed. The full evaluation includes failed courses.\n\nPlayback from the repository root:\n\n```bash\n{playback}\n```\n"
        (dest/'README.md').write_text(text)
        detail.append(f"### {spec['id']}: {spec['name']}\n\n**Method:** {spec['method']}.\n\n**Strengths:** "+' '.join(strength)+"\n\n**Weaknesses:** "+' '.join(weakness)+'\n')
        manifest={str(p.relative_to(dest)):sha(p) for p in sorted(dest.rglob('*')) if p.is_file() and p.name!='archive_manifest.json'}
        (dest/'archive_manifest.json').write_text(json.dumps({'algorithm':'sha256','files':manifest},indent=2)+'\n')
    # Include the observed strengths/weaknesses in the structured report too.
    (evidence/'all_model_results.json').write_text(json.dumps(summary,indent=2)+'\n')
    table=['| Model | Method | Completed / 100 | Mean test reward | Mean fruit | Falls | Timeouts |',
           '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for i,r in enumerate(records,1):
        s=r['stats']
        table.append(f"| {i}. {r['spec']['name']} | {r['spec']['method']} | {s['completions']} | {s['mean_reward']:.2f} | {s['mean_fruit']:.2f} | {s['falls']} | {s['timeouts']} |")
    if best_completion is best_reward:
        verdict=f"**Best overall in this benchmark: {best_completion['spec']['name']} ({best_completion['spec']['id']}).** It led both course completion ({best_completion['stats']['completions']}%) and mean standardized reward ({best_completion['stats']['mean_reward']:.2f})."
    else:
        verdict=f"**Best for finishing courses: {best_completion['spec']['name']} ({best_completion['stats']['completions']}%). Best for mean standardized reward: {best_reward['spec']['name']} ({best_reward['stats']['mean_reward']:.2f}).** There is no single model leading both metrics. For course completion as the primary objective, the recommendation is {best_completion['spec']['name']}; the reward tradeoff is shown below."
    if best_completion['spec']['curriculum']:
        selected_steps = best_completion['evaluation']['model_decisions']
        stages = best_completion['config']['curriculum_stages']
        selected_stage = next(stage for stage in reversed(stages) if stage['decisions'] <= selected_steps)
        if selected_stage['difficulty'] < 1:
            verdict += (f"\n\nThe winning checkpoint was saved at **{selected_steps:,} decisions**, during the difficulty "
                        f"**{selected_stage['difficulty']:.1f}** stage. Its training experience consisted of easier courses. "
                        "The run continued through the harder stages, but those later checkpoints did not replace it under the fixed validation rule. "
                        "The full-difficulty test result therefore supports the earlier easier-course policy; the contribution of later curriculum stages has not been demonstrated.")
    report=f"""# All-model efficiency study

{verdict}

The highest fruit collection was achieved by **{best_fruit['spec']['name']}**, at **{best_fruit['stats']['mean_fruit']:.2f} fruits per episode**. Fruit collection, training reward, and finishing the course are distinct outcomes.

## Matched benchmark results

Every selected policy was evaluated deterministically on the same 100 fresh full-difficulty randomized courses: **seeds 400000–400099**. The table uses this new shared test set for all seven models; it does not mix the earlier 20- and 100-course results.

"""+'\n'.join(table)+"""

![Completion and standardized reward](evidence/completion_and_reward.png)

![All test outcomes](evidence/all_outcomes.png)

## Methods and experimental controls

Models 1 and 2 reuse the original, preserved checkpoints. Models 3–6 each change one method relative to the 60-input gap-feature configuration. Model 7 combines all four further changes. No new model was initialized from the test-selected winner or tuned on the benchmark outcomes.

- **Action precision:** two physics ticks per decision instead of four.
- **Curriculum:** first 20% of training decisions use one missing floor tile and one guard; the next 30% use three missing tiles and two guards; the final 50% use the original five missing tiles and four guards. Physics, jumping rules, floating platforms, and full-difficulty test generation are unchanged.
- **Learning rate:** 0.0001 instead of 0.0003.
- **Fruit incentive:** training bonus +4 instead of +2; all final evaluation uses +2.
- **Combined:** two-tick decisions, curriculum, lower learning rate, and larger training fruit bonus, retaining the gap observations.

Each new run has a nominal budget of **2,000,000 physics ticks**: 500,000 decisions at four ticks or 1,000,000 decisions at two ticks. Episodes ending inside an action can slightly reduce actual ticks; the exact count is saved in each model's configuration. Both historical training searches also reached 500,000 four-tick decisions. These are comparable simulated-play budgets, not equal optimizer-update or wall-time budgets.

All new runs use seed 7, the same network architecture, replay-buffer size, per-decision discount, and exploration endpoints. New validation runs occur every 100,000 nominal ticks on seeds 100000–100004, always at full difficulty with the original reward formula. Checkpoints are chosen by validation completions, then mean forward progress, preserving the earlier selection rule. Test reward is not used to choose a checkpoint within a run. The best overall model is identified after the benchmark, so its reported result remains a descriptive benchmark selection rather than a fresh confirmation test.

The comparable test reward is +0.01 per new forward pixel, -0.01 per physics tick, +2 for a fruit-collection event, -2 for damage, +100 for finishing, and -10 for falling, defeat, or timeout. Raising the training fruit bonus cannot directly inflate these test scores.

## Individual strengths and weaknesses

"""+'\n'.join(detail)+"""
## Confidence and limits

This is one training seed per new method. The measured differences may include training randomness and historical baseline differences. They do not establish consistent superiority across repeated training runs. The original baseline was resumed from 25,000 decisions with a rebuilt replay buffer; the new models started from scratch. Its historical logs contain extra/resumed phases, so 500,000 is the final model decision counter rather than a strict accounting of all historical computation.

Two-tick variants receive twice as many decisions and potential optimizer updates for the same nominal simulation budget. Keeping gamma=0.995 per decision also changes the discount horizon in physics ticks. Their results therefore describe the full timing configuration, rather than isolating timing alone.

Only five validation layouts were used, so checkpoint selection can be noisy. One hundred matching test layouts improve comparability but do not cover every generated course. Approximate paired bootstrap 95% intervals for completion, reward, and fruit differences versus Models 1 and 2 are saved in the structured results. They describe course-sampling variation, not training-run variation, and are not corrected for multiple comparisons.

## Evidence and reproduction

Each model has its own folder containing selected and final weights, configuration, validation history, training logs/chart, full per-course evaluation, a gameplay GIF with a trajectory, console logs, a README, and SHA-256 checksums. Successful recordings are illustrative examples and are not substitutes for the full failure counts. All seven recordings were checked against their corresponding evaluation trajectory outcomes.

The plan was saved before training and testing in `plan.json`. The shared source snapshot is retained for reference; shared game assets remain in the main repository. `evidence/model_summary.csv` supports spreadsheets, `evidence/course_by_course.csv` contains all 700 outcomes, and `evidence/all_model_results.json` includes raw results and paired comparisons.

All pre-existing model artifacts were checked against their pre-study SHA-256 hashes and remain unchanged.
"""
    (folder/'REPORT.md').write_text(report)
    (folder/'README.md').write_text('# Seven-model efficiency study\n\nSee [the full report](REPORT.md), [summary spreadsheet](evidence/model_summary.csv), and each model folder for saved models and evidence. Original artifacts remain unchanged.\n')
    reports=PROJECT_ROOT/'reports'
    report_evidence=reports/'evidence/efficiency_study_2026_10_20'
    shutil.copytree(evidence,report_evidence,dirs_exist_ok=True)
    (reports/'All_Model_Comparison.md').write_text(report.replace('](evidence/','](evidence/efficiency_study_2026_10_20/'))
    shutil.copy2(Path(__file__),folder/'source_snapshot/agents/Deep_Q-Network'/Path(__file__).name)
    print(json.dumps({'best_completion':summary['best_completion'],'best_reward':summary['best_reward'],'best_fruit':summary['best_fruit'],
                      'original_files_preserved':len(preserved),
                      'models':{r['spec']['id']:r['stats'] for r in records}},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder',type=Path,required=True)
    args=parser.parse_args()
    summarize(args.folder)
