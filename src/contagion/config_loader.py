"""Config file loader for CLI tools.

Supports loading configuration from YAML or JSON files to reduce
command-line argument complexity for complex workflows.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML or JSON file.
    
    Args:
        config_path: Path to config file (.yaml, .yml, or .json)
        
    Returns:
        Dictionary of configuration parameters
        
    Raises:
        ValueError: If file extension is not supported
        FileNotFoundError: If config file doesn't exist
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    ext = path.suffix.lower()
    
    if ext == ".json":
        with open(path) as f:
            return json.load(f)
    elif ext in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError:
            raise ImportError(
                "PyYAML is required for YAML config files. "
                "Install with: pip install pyyaml"
            )
        with open(path) as f:
            return yaml.safe_load(f)
    else:
        raise ValueError(
            f"Unsupported config file extension: {ext}. "
            "Use .json, .yaml, or .yml"
        )


def merge_config_with_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    config_path: Optional[str] = None
) -> argparse.Namespace:
    """Merge config file with command-line arguments.
    
    Command-line arguments take precedence over config file values.
    
    Args:
        parser: Argument parser (used to get defaults)
        args: Parsed command-line arguments
        config_path: Optional path to config file
        
    Returns:
        Merged namespace with config and CLI args
    """
    if not config_path:
        return args
    
    config = load_config(config_path)
    
    # Get default values from parser
    defaults = {
        action.dest: action.default
        for action in parser._actions
        if action.dest != 'help'
    }
    
    # Merge: config < defaults < CLI args
    merged = {}
    
    # Start with config values
    for key, value in config.items():
        merged[key] = value
    
    # Override with non-default CLI args
    for key, value in vars(args).items():
        # If CLI arg is not default, use it (takes precedence)
        if key in defaults and value != defaults.get(key):
            merged[key] = value
        # If key not in config, use CLI value
        elif key not in config:
            merged[key] = value
    
    return argparse.Namespace(**merged)


def add_config_argument(parser: argparse.ArgumentParser) -> None:
    """Add --config argument to parser.
    
    Args:
        parser: Argument parser to add --config to
    """
    parser.add_argument(
        "--config",
        type=str,
        help="Path to config file (JSON or YAML) - CLI args override config values"
    )
