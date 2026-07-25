import yaml
from pathlib import Path


def load_config(path="config/config.yaml"):
    with open(Path(__file__).parent.parent / path) as f:
        return yaml.safe_load(f)
