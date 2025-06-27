import yaml
from pathlib import Path

CONFIG_PATH = Path("configs")

def load_model_config():
    with open(CONFIG_PATH / "models.yaml", 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_keyword_config():
    with open(CONFIG_PATH / "keywords.yaml", 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)