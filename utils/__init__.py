"""
Utils package for Warehouse Layout Generator.

This package contains utility modules and functions used throughout the application.
Key functionality includes geometry operations, input validation, and common helper functions.
"""

from utils.geometry import (
    calculate_distance,
    calculate_area, 
    calculate_perimeter,
    point_in_polygon,
    calculate_bounding_box,
    rotate_point
)

from utils.validators import (
    validate_dimensions,
    validate_numeric,
    validate_polygon,
    validate_config
)

from utils.helpers import (
    format_area,
    format_distance,
    create_unique_id,
    load_json,
    save_json,
    timer_decorator
)

__all__ = [
    # Geometry functions
    'calculate_distance',
    'calculate_area',
    'calculate_perimeter',
    'point_in_polygon',
    'calculate_bounding_box',
    'rotate_point',
    
    # Validator functions
    'validate_dimensions',
    'validate_numeric',
    'validate_polygon',
    'validate_config',
    
    # Helper functions
    'format_area',
    'format_distance',
    'create_unique_id',
    'load_json',
    'save_json',
    'timer_decorator'
]