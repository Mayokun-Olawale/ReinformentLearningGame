"""Load the decision cadence saved with a policy."""
import json
import argparse
from pathlib import Path


def action_repeat_for(model_path):
    parent = Path(model_path).resolve().parent
    for candidate in (parent / "config.json", parent.parent / "config.json"):
        if candidate.exists():
            repeat = int(json.loads(candidate.read_text()).get("action_repeat", 1))
            if repeat not in (1, 4):
                raise ValueError(f"Unsupported saved action_repeat: {repeat}")
            return repeat
    return 1


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number
