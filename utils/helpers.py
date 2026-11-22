import os
import sys
import logging
import json
from typing import Any, Dict, Optional

# Configure logging for the application
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

def check_file_exists(file_path: str) -> bool:
    """
    Checks if a file exists at the specified path.

    Args:
        file_path (str): The path to the file.

    Returns:
        bool: True if the file exists and is a file, False otherwise.
    """
    exists = os.path.isfile(file_path)
    if not exists:
        logger.warning(f"File not found: {file_path}")
    return exists

def load_json_config(config_path: str) -> Optional[Dict[str, Any]]:
    """
    Loads configuration data from a JSON file.

    Args:
        config_path (str): The path to the JSON configuration file.

    Returns:
        Optional[Dict[str, Any]]: A dictionary containing the config data,
                                   or None if loading fails.
    """
    if not check_file_exists(config_path):
        return None

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            logger.info(f"Successfully loaded configuration from {config_path}")
            return data
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON from {config_path}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error reading configuration file {config_path}: {e}")
        return None

def process_raw_text(text: str) -> str:
    """
    Processes raw input text by stripping whitespace and normalizing it.

    Args:
        text (str): The input text.

    Returns:
        str: The processed text.
    """
    if not text:
        return ""
    processed = text.strip()
    return processed

if __name__ == "__main__":
    # Self-test demonstration for the utils file
    print("Running self-test for utils/helpers.py")
    test_text = "  Hello Portfolio Project  "
    print(f"Original: '{test_text}'")
    print(f"Processed: '{process_raw_text(test_text)}'")
    
    config_exists = check_file_exists("config.json")
    print(f"config.json exists: {config_exists}")