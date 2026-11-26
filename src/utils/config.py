from pathlib import Path
import yaml

BASE_DIR = Path(__file__).resolve().parent.parent.parent

def load_config(folder: str="config", file: str="config.yaml") -> dict:
    """
    Load configuration from YAML file.

    args:
        folder(str): The folder whre configuration file is located.
        file(str): The name of configuration file.
    
    returns:
        dict: A dictionary containing the configuration parameteres.
    """

    config_path = BASE_DIR / folder / file

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    for key, relative_path in config["paths"].items():
        absolute_path = BASE_DIR / relative_path
        config["paths"][key] = str(absolute_path)

    return config
