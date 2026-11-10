"""Plot PPO diagnostics and validation completion against samples and time."""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot(run_dir):
    folder = Path(run_dir)
    with (folder/'metrics.csv').open() as stream:
        metrics = list(csv.DictReader(stream))
    validation = json.loads((folder/'validation.json').read_text())
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    steps = [int(row['steps']) for row in metrics]
    axes[0, 0].plot(steps, [float(row['value_loss']) for row in metrics])
    axes[0, 0].set(xlabel='Environment transitions', ylabel='Value loss')
    axes[0, 1].plot(steps, [float(row['entropy']) for row in metrics])
    axes[0, 1].set(xlabel='Environment transitions', ylabel='Policy entropy')
    for axis, key, label in [(axes[1, 0], 'steps', 'Environment transitions'),
                              (axes[1, 1], 'elapsed_seconds', 'Elapsed seconds')]:
        axis.plot([row[key] for row in validation],
                  [row['summary']['completion_rate'] for row in validation], marker='o')
        axis.set(xlabel=label, ylabel='Validation completion rate', ylim=(-.05, 1.05))
    fig.tight_layout()
    path = folder/'training.png'
    fig.savefig(path)
    plt.close(fig)
    return path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    print(plot(parser.parse_args().run_dir))
