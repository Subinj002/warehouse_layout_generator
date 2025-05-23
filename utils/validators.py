"""
Validation utilities for the Warehouse Layout Generator.

This module contains functions to validate input data, configurations,
and layout elements to ensure they meet required criteria before processing.
"""
import json
import os
import re
from typing import Dict, List, Tuple, Any, Optional, Union


def validate_dimensions(width: float, length: float, height: Optional[float] = None) -> bool:
    """
    Validate that dimensions are positive numbers.
    
    Args:
        width: Width value to validate
        length: Length value to validate
        height: Optional height value to validate
        
    Returns:
        bool: True if all dimensions are valid
        
    Raises:
        ValueError: If any dimension is invalid
    """
    if width <= 0:
        raise ValueError(f"Width must be a positive number, got {width}")
    if length <= 0:
        raise ValueError(f"Length must be a positive number, got {length}")
    if height is not None and height <= 0:
        raise ValueError(f"Height must be a positive number, got {height}")
    return True


def validate_coordinates(x: float, y: float, z: Optional[float] = None) -> bool:
    """
    Validate that coordinates are valid numbers.
    
    Args:
        x: X-coordinate to validate
        y: Y-coordinate to validate
        z: Optional Z-coordinate to validate
        
    Returns:
        bool: True if all coordinates are valid
    """
    # Coordinates can be negative, but must be numbers
    if not isinstance(x, (int, float)):
        raise ValueError(f"X-coordinate must be a number, got {type(x)}")
    if not isinstance(y, (int, float)):
        raise ValueError(f"Y-coordinate must be a number, got {type(y)}")
    if z is not None and not isinstance(z, (int, float)):
        raise ValueError(f"Z-coordinate must be a number, got {type(z)}")
    return True


def validate_angle(angle: float) -> bool:
    """
    Validate that an angle is between 0 and 360 degrees.
    
    Args:
        angle: Angle in degrees to validate
        
    Returns:
        bool: True if angle is valid
        
    Raises:
        ValueError: If angle is invalid
    """
    if not 0 <= angle < 360:
        raise ValueError(f"Angle must be between 0 and 360 degrees, got {angle}")
    return True


def validate_percentage(value: float, name: str = "Value") -> bool:
    """
    Validate that a value is a percentage between 0 and 100.
    
    Args:
        value: Percentage value to validate
        name: Name of the value for error message
        
    Returns:
        bool: True if percentage is valid
        
    Raises:
        ValueError: If percentage is invalid
    """
    if not 0 <= value <= 100:
        raise ValueError(f"{name} must be between 0 and 100 percent, got {value}")
    return True


def validate_config_file(file_path: str) -> Dict:
    """
    Validate that a configuration file exists and contains valid JSON.
    
    Args:
        file_path: Path to the configuration file
        
    Returns:
        Dict: Parsed configuration dictionary
        
    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If file contains invalid JSON
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    
    try:
        with open(file_path, 'r') as f:
            config = json.load(f)
            
        if not isinstance(config, dict):
            raise ValueError(f"Configuration must be a dictionary, got {type(config)}")
            
        return config
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in configuration file: {e}")


def validate_warehouse_boundaries(width: float, length: float, elements: List[Dict]) -> bool:
    """
    Validate that all warehouse elements fit within the warehouse boundaries.
    
    Args:
        width: Warehouse width
        length: Warehouse length
        elements: List of warehouse elements with coordinates and dimensions
        
    Returns:
        bool: True if all elements fit within boundaries
        
    Raises:
        ValueError: If any element exceeds boundaries
    """
    for i, element in enumerate(elements):
        if element.get('type') == 'IGNORE':
            continue
            
        # Get element position and dimensions
        x = element.get('x', 0)
        y = element.get('y', 0)
        element_width = element.get('width', 0)
        element_length = element.get('length', 0)
        
        # Account for rotation if present
        rotation = element.get('rotation', 0)
        if rotation == 90 or rotation == 270:
            element_width, element_length = element_length, element_width
        
        # Check if element exceeds boundaries
        if x < 0 or y < 0 or x + element_width > width or y + element_length > length:
            raise ValueError(
                f"Element {i} (type: {element.get('type')}) exceeds warehouse boundaries. "
                f"Position: ({x}, {y}), Size: {element_width}x{element_length}, "
                f"Warehouse size: {width}x{length}"
            )
    
    return True


def validate_element_overlap(elements: List[Dict], tolerance: float = 0.01) -> bool:
    """
    Validate that warehouse elements do not overlap.
    
    Args:
        elements: List of warehouse elements with coordinates and dimensions
        tolerance: Allowed overlap tolerance (for numerical precision issues)
        
    Returns:
        bool: True if no elements overlap
        
    Raises:
        ValueError: If any elements overlap
    """
    for i, element1 in enumerate(elements):
        if element1.get('type') == 'IGNORE':
            continue
            
        # Get element1 position and dimensions
        x1 = element1.get('x', 0)
        y1 = element1.get('y', 0)
        w1 = element1.get('width', 0)
        l1 = element1.get('length', 0)
        
        # Account for rotation if present
        rot1 = element1.get('rotation', 0)
        if rot1 == 90 or rot1 == 270:
            w1, l1 = l1, w1
        
        for j, element2 in enumerate(elements[i+1:], i+1):
            if element2.get('type') == 'IGNORE':
                continue
                
            # Get element2 position and dimensions
            x2 = element2.get('x', 0)
            y2 = element2.get('y', 0)
            w2 = element2.get('width', 0)
            l2 = element2.get('length', 0)
            
            # Account for rotation if present
            rot2 = element2.get('rotation', 0)
            if rot2 == 90 or rot2 == 270:
                w2, l2 = l2, w2
            
            # Check for overlap
            overlap_x = (x1 < x2 + w2 - tolerance) and (x2 < x1 + w1 - tolerance)
            overlap_y = (y1 < y2 + l2 - tolerance) and (y2 < y1 + l1 - tolerance)
            
            if overlap_x and overlap_y:
                raise ValueError(
                    f"Elements {i} and {j} overlap. "
                    f"Element {i}: Position ({x1}, {y1}), Size {w1}x{l1}. "
                    f"Element {j}: Position ({x2}, {y2}), Size {w2}x{l2}."
                )
    
    return True


def validate_clearance(elements: List[Dict], clearance_requirements: Dict[str, Dict[str, float]]) -> bool:
    """
    Validate that elements maintain required clearance from each other.
    
    Args:
        elements: List of warehouse elements with coordinates and dimensions
        clearance_requirements: Dictionary mapping element types to required clearances
        
    Returns:
        bool: True if all clearance requirements are met
        
    Raises:
        ValueError: If clearance requirements are not met
    """
    for i, element1 in enumerate(elements):
        element1_type = element1.get('type')
        if element1_type == 'IGNORE':
            continue
            
        # Get element1 position and dimensions
        x1 = element1.get('x', 0)
        y1 = element1.get('y', 0)
        w1 = element1.get('width', 0)
        l1 = element1.get('length', 0)
        
        # Account for rotation if present
        rot1 = element1.get('rotation', 0)
        if rot1 == 90 or rot1 == 270:
            w1, l1 = l1, w1
        
        for j, element2 in enumerate(elements):
            if i == j or element2.get('type') == 'IGNORE':
                continue
                
            element2_type = element2.get('type')
                
            # Get required clearance between these element types
            required_clearance = clearance_requirements.get(element1_type, {}).get(element2_type, 0)
            if required_clearance == 0:
                continue
                
            # Get element2 position and dimensions
            x2 = element2.get('x', 0)
            y2 = element2.get('y', 0)
            w2 = element2.get('width', 0)
            l2 = element2.get('length', 0)
            
            # Account for rotation if present
            rot2 = element2.get('rotation', 0)
            if rot2 == 90 or rot2 == 270:
                w2, l2 = l2, w2
            
            # Calculate distances between elements
            distance_x = max(0, min(abs(x1 - (x2 + w2)), abs(x2 - (x1 + w1))))
            distance_y = max(0, min(abs(y1 - (y2 + l2)), abs(y2 - (y1 + l1))))
            
            # If elements are not overlapping in one dimension, use Euclidean distance
            if distance_x > 0 and distance_y > 0:
                distance = (distance_x**2 + distance_y**2)**0.5
            else:
                # If elements are aligned in one dimension, use Manhattan distance
                distance = distance_x + distance_y
            
            if distance < required_clearance:
                raise ValueError(
                    f"Clearance requirement not met between elements {i} ({element1_type}) and {j} ({element2_type}). "
                    f"Required clearance: {required_clearance}, Actual distance: {distance:.2f}"
                )
    
    return True


def validate_file_path(file_path: str, must_exist: bool = False, file_type: str = None) -> bool:
    """
    Validate a file path.
    
    Args:
        file_path: Path to validate
        must_exist: If True, file must exist
        file_type: Optional file extension to check (without the dot)
        
    Returns:
        bool: True if path is valid
        
    Raises:
        ValueError: If path is invalid
    """
    if not isinstance(file_path, str):
        raise ValueError(f"File path must be a string, got {type(file_path)}")
        
    if must_exist and not os.path.exists(file_path):
        raise ValueError(f"File does not exist: {file_path}")
        
    if file_type:
        if not file_path.lower().endswith(f".{file_type.lower()}"):
            raise ValueError(f"File must be a {file_type} file: {file_path}")
            
    return True


def validate_name(name: str, pattern: str = r'^[A-Za-z0-9_\-\s]+$') -> bool:
    """
    Validate that a name contains only allowed characters.
    
    Args:
        name: Name to validate
        pattern: Regex pattern for allowed characters
        
    Returns:
        bool: True if name is valid
        
    Raises:
        ValueError: If name is invalid
    """
    if not name:
        raise ValueError("Name cannot be empty")
        
    if not re.match(pattern, name):
        raise ValueError(
            f"Name contains invalid characters: {name}. "
            f"Only letters, numbers, underscores, hyphens, and spaces are allowed."
        )
        
    return True


def validate_constraints(elements: List[Dict], constraints: List[Dict]) -> bool:
    """
    Validate that all layout constraints are satisfied.
    
    Args:
        elements: List of warehouse elements
        constraints: List of constraint dictionaries
        
    Returns:
        bool: True if all constraints are satisfied
        
    Raises:
        ValueError: If any constraint is not satisfied
    """
    for i, constraint in enumerate(constraints):
        constraint_type = constraint.get('type')
        
        if constraint_type == 'alignment':
            # Alignment constraint checks
            element_ids = constraint.get('element_ids', [])
            alignment = constraint.get('alignment')  # 'horizontal' or 'vertical'
            
            if alignment == 'horizontal':
                y_values = [elements[eid].get('y') for eid in element_ids if eid < len(elements)]
                if len(set(y_values)) > 1:
                    raise ValueError(f"Horizontal alignment constraint {i} not satisfied")
                    
            elif alignment == 'vertical':
                x_values = [elements[eid].get('x') for eid in element_ids if eid < len(elements)]
                if len(set(x_values)) > 1:
                    raise ValueError(f"Vertical alignment constraint {i} not satisfied")
        
        elif constraint_type == 'distance':
            # Distance constraint checks
            element1_id = constraint.get('element1_id')
            element2_id = constraint.get('element2_id')
            min_distance = constraint.get('min_distance', 0)
            max_distance = constraint.get('max_distance', float('inf'))
            
            if element1_id >= len(elements) or element2_id >= len(elements):
                continue
                
            element1 = elements[element1_id]
            element2 = elements[element2_id]
            
            # Calculate center points
            x1 = element1.get('x', 0) + element1.get('width', 0) / 2
            y1 = element1.get('y', 0) + element1.get('length', 0) / 2
            x2 = element2.get('x', 0) + element2.get('width', 0) / 2
            y2 = element2.get('y', 0) + element2.get('length', 0) / 2
            
            # Calculate distance
            distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
            
            if distance < min_distance or distance > max_distance:
                raise ValueError(
                    f"Distance constraint {i} not satisfied. "
                    f"Required: {min_distance}-{max_distance}, Actual: {distance:.2f}"
                )
        
        # Add more constraint types as needed
    
    return True


def validate_aisle_width(elements: List[Dict], min_aisle_width: float) -> bool:
    """
    Validate that aisles meet minimum width requirements.
    
    Args:
        elements: List of warehouse elements
        min_aisle_width: Minimum required aisle width
        
    Returns:
        bool: True if all aisles meet width requirements
        
    Raises:
        ValueError: If any aisle is too narrow
    """
    # This is a simplified version - a real implementation would need to
    # identify actual aisles and measure their widths based on the layout
    
    # For now, we'll just check shelves that are oriented in the same direction
    shelves = [e for e in elements if e.get('type', '').lower() in ('shelf', 'rack', 'shelving')]
    
    for i, shelf1 in enumerate(shelves):
        x1 = shelf1.get('x', 0)
        y1 = shelf1.get('y', 0)
        w1 = shelf1.get('width', 0)
        l1 = shelf1.get('length', 0)
        rotation1 = shelf1.get('rotation', 0)
        
        # Standardize dimensions based on rotation
        if rotation1 == 90 or rotation1 == 270:
            w1, l1 = l1, w1
        
        for j, shelf2 in enumerate(shelves[i+1:], i+1):
            x2 = shelf2.get('x', 0)
            y2 = shelf2.get('y', 0)
            w2 = shelf2.get('width', 0)
            l2 = shelf2.get('length', 0)
            rotation2 = shelf2.get('rotation', 0)
            
            # Standardize dimensions based on rotation
            if rotation2 == 90 or rotation2 == 270:
                w2, l2 = l2, w2
            
            # Only check shelves with the same orientation
            if rotation1 % 180 == rotation2 % 180:
                # Check if shelves are parallel to each other
                # For vertical shelves (rotation 0 or 180)
                if rotation1 % 180 == 0:
                    overlap_y = min(y1 + l1, y2 + l2) - max(y1, y2)
                    if overlap_y > 0:
                        distance = abs((x1 + w1) - x2) if x1 < x2 else abs((x2 + w2) - x1)
                        if 0 < distance < min_aisle_width:
                            raise ValueError(
                                f"Aisle between shelves {i} and {j} is too narrow. "
                                f"Required: {min_aisle_width}, Actual: {distance:.2f}"
                            )
                # For horizontal shelves (rotation 90 or 270)
                else:
                    overlap_x = min(x1 + w1, x2 + w2) - max(x1, x2)
                    if overlap_x > 0:
                        distance = abs((y1 + l1) - y2) if y1 < y2 else abs((y2 + l2) - y1)
                        if 0 < distance < min_aisle_width:
                            raise ValueError(
                                f"Aisle between shelves {i} and {j} is too narrow. "
                                f"Required: {min_aisle_width}, Actual: {distance:.2f}"
                            )
    
    return True


def validate_warehouse_layout(layout: Dict) -> bool:
    """
    Comprehensive validation of a warehouse layout.
    
    Args:
        layout: Dictionary with complete warehouse layout information
        
    Returns:
        bool: True if layout is valid
        
    Raises:
        ValueError: If layout is invalid
    """
    # Validate basic layout properties
    validate_dimensions(layout.get('width', 0), layout.get('length', 0))
    
    # Validate elements
    elements = layout.get('elements', [])
    if not elements:
        raise ValueError("Layout must contain at least one element")
    
    # Validate each element
    for i, element in enumerate(elements):
        element_type = element.get('type')
        if not element_type:
            raise ValueError(f"Element {i} is missing a type")
        
        # Skip validation for annotation or reference elements
        if element_type.lower() in ('text', 'label', 'note', 'reference', 'ignore'):
            continue
        
        # Validate element dimensions
        validate_dimensions(
            element.get('width', 0), 
            element.get('length', 0),
            element.get('height', None)
        )
        
        # Validate element position
        validate_coordinates(element.get('x', 0), element.get('y', 0))
        
        # Validate element rotation if present
        if 'rotation' in element:
            validate_angle(element['rotation'])
    
    # Validate that elements are within warehouse boundaries
    validate_warehouse_boundaries(layout.get('width', 0), layout.get('length', 0), elements)
    
    # Validate that elements don't overlap
    validate_element_overlap(elements)
    
    # Additional validations can be added here
    
    return True