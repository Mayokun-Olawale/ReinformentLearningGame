"""Paths shared by training tools, independent of the working directory."""
import sys
from pathlib import Path

TRAINING_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TRAINING_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
LOG_DIR = TRAINING_DIR / "tensorboard_logs"
MODEL_DIR = TRAINING_DIR / "saved_models"
CHART_PATH = TRAINING_DIR / "training_results.png"
