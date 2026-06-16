import json
import os

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")


def load_config() -> dict:
    """
    Load project configurations from core/config.json with default fallbacks.

    Returns:
        dict: The config dictionary.
    """
    default_config = {
        "nlu": {
            "embedding_model": "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
            "thresh_evidence": 0.40,
            "thresh_value": 0.55
        },
        "parser": {
            "model_name": "openai/gpt-oss-safeguard-20b",
            "fallback_models": [],
            "max_retries": 3,
            "retry_delay": 1.0,
            "request_timeout": 30.0
        },
        "traversal": {
            "smooth": 1e-06,
            "max_delta": 2.0,
            "absence_prob_threshold": 0.5,
            "absence_weight": 0.5
        }
    }

    if not os.path.exists(CONFIG_PATH):
        return default_config

    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            user_config = json.load(f)
            # Merge user config over default config keys
            for key, val in default_config.items():
                if key in user_config:
                    default_config[key].update(user_config[key])
            return default_config
    except Exception as e:
        print(f"Warning: Failed to load config.json ({e}). Using defaults.")
        return default_config
