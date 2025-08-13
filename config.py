from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent # Project root (automatically calculated from the config file location)

# Configurable sub-paths
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

