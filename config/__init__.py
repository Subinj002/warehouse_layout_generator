"""
Configuration module for Warehouse Layout Generator.

This module provides access to configuration settings throughout the application.
It handles loading, validating, and providing access to configuration parameters
from default settings and user-provided configuration files.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional

from .config_schema import validate_config

# Default configuration file path
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "default_config.json")

# Singleton config instance
_config_instance = None


def get_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Get the application configuration.
    
    This function returns the configuration as a dictionary, loading it from
    the specified path or using the default configuration if no path is provided.
    
    Args:
        config_path: Optional path to a JSON configuration file.
                    If None, uses the default configuration.
    
    Returns:
        Dict[str, Any]: The configuration dictionary
    
    Raises:
        FileNotFoundError: If the configuration file doesn't exist
        ValueError: If the configuration is invalid
    """
    global _config_instance
    
    if _config_instance is None:
        _config_instance = _load_config(config_path)
    
    return _config_instance


def _load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from a file.
    
    Args:
        config_path: Path to the configuration file. If None, uses default config.
    
    Returns:
        Dict[str, Any]: The loaded and validated configuration
    
    Raises:
        FileNotFoundError: If the configuration file doesn't exist
        ValueError: If the configuration is invalid
    """
    # Load default configuration
    with open(DEFAULT_CONFIG_PATH, 'r') as f:
        config = json.load(f)
    
    # Override with user config if provided
    if config_path is not None:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            user_config = json.load(f)
            
        # Merge user config with default config
        _deep_update(config, user_config)
    
    # Validate configuration
    validate_config(config)
    
    return config


def _deep_update(base_dict: Dict[str, Any], update_dict: Dict[str, Any]) -> None:
    """
    Recursively update a dictionary with another dictionary.
    
    Args:
        base_dict: The dictionary to update
        update_dict: The dictionary containing updates
    """
    for key, value in update_dict.items():
        if (
            key in base_dict and 
            isinstance(base_dict[key], dict) and 
            isinstance(value, dict)
        ):
            _deep_update(base_dict[key], value)
        else:
            base_dict[key] = value


def reset_config() -> None:
    """
    Reset the configuration to force reloading on next access.
    This is primarily useful for testing.
    """
    global _config_instance
    _config_instance = None