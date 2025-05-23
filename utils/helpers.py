"""
Utility helper functions for the Warehouse Layout Generator.

This module provides various helper functions used throughout the application for:
- File operations
- Unit conversions
- Logging utilities
- Configuration management
- Data validation helpers
- Error handling utilities
"""

import os
import json
import logging
import math
import time
import hashlib
from typing import Dict, List, Tuple, Union, Any, Optional

# Set up logging
logger = logging.getLogger(__name__)

# Constants
METERS_TO_FEET = 3.28084
FEET_TO_METERS = 0.3048
INCHES_TO_MM = 25.4
MM_TO_INCHES = 0.0393701
DEFAULT_PRECISION = 2


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> None:
    """
    Set up logging configuration for the application.
    
    Args:
        log_level: The logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to a log file
    """
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Convert string log level to actual level
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    
    # Configure logging
    handlers = []
    handlers.append(logging.StreamHandler())
    
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=numeric_level,
        format=log_format,
        handlers=handlers
    )
    
    logger.info(f"Logging initialized at {log_level} level")


def load_json_file(file_path: str) -> Dict:
    """
    Load and parse a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Dict containing the parsed JSON data
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the file contains invalid JSON
    """
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in file {file_path}: {str(e)}")
        raise


def save_json_file(data: Dict, file_path: str, pretty: bool = True) -> None:
    """
    Save data to a JSON file.
    
    Args:
        data: Dictionary to save as JSON
        file_path: Path where to save the file
        pretty: If True, format the JSON with indentation
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w') as file:
            if pretty:
                json.dump(data, file, indent=4)
            else:
                json.dump(data, file)
        
        logger.debug(f"Data successfully saved to {file_path}")
    except Exception as e:
        logger.error(f"Error saving JSON to {file_path}: {str(e)}")
        raise


def format_dimensions(value: float, unit: str = "m", precision: int = DEFAULT_PRECISION) -> str:
    """
    Format a dimension value with the appropriate unit.
    
    Args:
        value: The dimension value to format
        unit: The unit of measurement (m, ft, mm, etc.)
        precision: Number of decimal places
        
    Returns:
        Formatted string with value and unit
    """
    return f"{value:.{precision}f} {unit}"


def convert_units(value: float, from_unit: str, to_unit: str) -> float:
    """
    Convert a value between different units of measurement.
    
    Args:
        value: The value to convert
        from_unit: Source unit (m, ft, mm, in)
        to_unit: Target unit (m, ft, mm, in)
        
    Returns:
        The converted value
        
    Raises:
        ValueError: If the unit conversion is not supported
    """
    # Convert to meters first (our base unit)
    if from_unit == "m":
        meters = value
    elif from_unit == "ft":
        meters = value * FEET_TO_METERS
    elif from_unit == "mm":
        meters = value / 1000.0
    elif from_unit == "in":
        meters = value * INCHES_TO_MM / 1000.0
    else:
        raise ValueError(f"Unsupported source unit: {from_unit}")
    
    # Convert from meters to target unit
    if to_unit == "m":
        return meters
    elif to_unit == "ft":
        return meters * METERS_TO_FEET
    elif to_unit == "mm":
        return meters * 1000.0
    elif to_unit == "in":
        return meters * 1000.0 / INCHES_TO_MM
    else:
        raise ValueError(f"Unsupported target unit: {to_unit}")


def calculate_area(length: float, width: float) -> float:
    """
    Calculate the area of a rectangle.
    
    Args:
        length: Length of the rectangle
        width: Width of the rectangle
        
    Returns:
        Area of the rectangle
    """
    return length * width


def calculate_utilization(used_area: float, total_area: float) -> float:
    """
    Calculate space utilization percentage.
    
    Args:
        used_area: The area being utilized
        total_area: The total available area
        
    Returns:
        Utilization as a percentage (0-100)
    """
    if total_area <= 0:
        return 0.0
    
    utilization = (used_area / total_area) * 100.0
    return min(utilization, 100.0)  # Cap at 100%


def generate_unique_id(prefix: str = "") -> str:
    """
    Generate a unique ID based on timestamp and random data.
    
    Args:
        prefix: Optional prefix for the ID
        
    Returns:
        A unique ID string
    """
    timestamp = str(time.time()).encode('utf-8')
    hash_obj = hashlib.md5(timestamp)
    unique_id = hash_obj.hexdigest()[:12]
    
    if prefix:
        return f"{prefix}_{unique_id}"
    return unique_id


def get_project_root() -> str:
    """
    Get the project root directory.
    
    Returns:
        Absolute path to the project root directory
    """
    # This assumes the helpers.py file is in the utils directory under the project root
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(current_dir)


def ensure_directory_exists(directory_path: str) -> None:
    """
    Ensure that a directory exists, creating it if necessary.
    
    Args:
        directory_path: Path to the directory
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        logger.debug(f"Created directory: {directory_path}")


def is_valid_file_path(file_path: str) -> bool:
    """
    Check if a file path is valid and writable.
    
    Args:
        file_path: Path to check
        
    Returns:
        True if the path is valid and writable, False otherwise
    """
    try:
        directory = os.path.dirname(file_path) or '.'
        return os.access(directory, os.W_OK)
    except Exception:
        return False


def timer_decorator(func):
    """
    Decorator to measure and log the execution time of a function.
    
    Args:
        func: The function to time
        
    Returns:
        Wrapped function that logs execution time
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.debug(f"Function {func.__name__} executed in {end_time - start_time:.4f} seconds")
        return result
    return wrapper


def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, handling division by zero.
    
    Args:
        a: Numerator
        b: Denominator
        default: Default value to return if denominator is zero
        
    Returns:
        Result of division or default value
    """
    return a / b if b != 0 else default


def flatten_dict(d: Dict, parent_key: str = '', sep: str = '.') -> Dict:
    """
    Flatten a nested dictionary structure.
    
    Args:
        d: Dictionary to flatten
        parent_key: Base key for the current level
        sep: Separator between nested keys
        
    Returns:
        Flattened dictionary
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def chunk_list(lst: List, chunk_size: int) -> List[List]:
    """
    Split a list into chunks of specified size.
    
    Args:
        lst: List to split
        chunk_size: Size of each chunk
        
    Returns:
        List of chunks
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def deep_update(original: Dict, update: Dict) -> Dict:
    """
    Deep update a nested dictionary structure.
    
    Args:
        original: Original dictionary to update
        update: Dictionary with updates to apply
        
    Returns:
        Updated dictionary
    """
    for key, value in update.items():
        if isinstance(value, dict) and key in original and isinstance(original[key], dict):
            original[key] = deep_update(original[key], value)
        else:
            original[key] = value
    return original


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename by removing invalid characters.
    
    Args:
        filename: The filename to sanitize
        
    Returns:
        Sanitized filename
    """
    # Replace invalid characters with underscores
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    return filename