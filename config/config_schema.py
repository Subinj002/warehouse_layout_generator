"""
Configuration schema for the Warehouse Layout Generator.

This module defines the schema for validating warehouse configuration files,
including constraints, dimensions, and element specifications.
"""

import json
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field


@dataclass
class DimensionConstraints:
    """Constraints for warehouse dimensions."""
    min_width: float = 10.0
    max_width: float = 500.0
    min_length: float = 10.0
    max_length: float = 500.0
    min_height: float = 3.0
    max_height: float = 15.0
    

@dataclass
class ElementConstraints:
    """Constraints for warehouse elements."""
    min_aisle_width: float = 1.5
    max_aisle_width: float = 5.0
    min_rack_length: float = 1.0
    max_rack_length: float = 50.0
    min_rack_width: float = 0.6
    max_rack_width: float = 2.0
    min_rack_height: float = 1.0
    max_rack_height: float = 12.0
    min_staging_area: float = 10.0
    max_staging_area: float = 200.0
    

@dataclass
class SafetyConstraints:
    """Safety and regulatory constraints."""
    fire_exit_min_width: float = 1.2
    fire_exit_max_distance: float = 50.0
    emergency_path_min_width: float = 1.0
    sprinkler_coverage_radius: float = 3.0
    min_distance_between_racks: float = 0.15


@dataclass
class RackSpecification:
    """Specification for a storage rack."""
    type: str
    width: float
    length: float
    height: float
    load_capacity: float
    shelf_count: int
    clearance_required: float
    

@dataclass
class AisleSpecification:
    """Specification for an aisle."""
    type: str
    width: float
    direction: str  # 'horizontal', 'vertical'
    

@dataclass
class ZoneSpecification:
    """Specification for a warehouse zone."""
    name: str
    type: str
    priority: int
    min_area: float
    max_area: Optional[float] = None
    adjacency: List[str] = field(default_factory=list)


@dataclass
class WarehouseConfig:
    """Full warehouse configuration."""
    name: str
    width: float
    length: float
    height: float
    unit: str = "m"  # Default unit is meters
    racks: List[RackSpecification] = field(default_factory=list)
    aisles: List[AisleSpecification] = field(default_factory=list)
    zones: List[ZoneSpecification] = field(default_factory=list)
    dimension_constraints: DimensionConstraints = field(default_factory=DimensionConstraints)
    element_constraints: ElementConstraints = field(default_factory=ElementConstraints)
    safety_constraints: SafetyConstraints = field(default_factory=SafetyConstraints)
    optimization_goals: List[str] = field(default_factory=lambda: ["space_utilization"])
    

class ConfigValidator:
    """Validates warehouse configurations against the schema."""
    
    def __init__(self):
        self.errors = []
        
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate the provided configuration dictionary against the schema.
        
        Args:
            config: Dictionary containing warehouse configuration
            
        Returns:
            bool: True if configuration is valid, False otherwise
        """
        self.errors = []
        
        # Check required top-level fields
        required_fields = ["name", "width", "length", "height"]
        for field in required_fields:
            if field not in config:
                self.errors.append(f"Missing required field: {field}")
        
        if self.errors:
            return False
            
        # Validate dimensions
        dimension_constraints = DimensionConstraints()
        if not (dimension_constraints.min_width <= config["width"] <= dimension_constraints.max_width):
            self.errors.append(f"Width must be between {dimension_constraints.min_width} and {dimension_constraints.max_width}")
            
        if not (dimension_constraints.min_length <= config["length"] <= dimension_constraints.max_length):
            self.errors.append(f"Length must be between {dimension_constraints.min_length} and {dimension_constraints.max_length}")
            
        if not (dimension_constraints.min_height <= config["height"] <= dimension_constraints.max_height):
            self.errors.append(f"Height must be between {dimension_constraints.min_height} and {dimension_constraints.max_height}")
        
        # Validate racks if present
        if "racks" in config:
            self._validate_racks(config["racks"])
            
        # Validate aisles if present
        if "aisles" in config:
            self._validate_aisles(config["aisles"])
            
        # Validate zones if present
        if "zones" in config:
            self._validate_zones(config["zones"])
            
        return len(self.errors) == 0
    
    def _validate_racks(self, racks: List[Dict[str, Any]]) -> None:
        """Validate rack specifications."""
        element_constraints = ElementConstraints()
        
        for i, rack in enumerate(racks):
            if "type" not in rack:
                self.errors.append(f"Rack {i}: Missing 'type' field")
                
            if "width" not in rack:
                self.errors.append(f"Rack {i}: Missing 'width' field")
            elif not (element_constraints.min_rack_width <= rack["width"] <= element_constraints.max_rack_width):
                self.errors.append(f"Rack {i}: Width must be between {element_constraints.min_rack_width} and {element_constraints.max_rack_width}")
                
            if "length" not in rack:
                self.errors.append(f"Rack {i}: Missing 'length' field")
            elif not (element_constraints.min_rack_length <= rack["length"] <= element_constraints.max_rack_length):
                self.errors.append(f"Rack {i}: Length must be between {element_constraints.min_rack_length} and {element_constraints.max_rack_length}")
                
            if "height" not in rack:
                self.errors.append(f"Rack {i}: Missing 'height' field")
            elif not (element_constraints.min_rack_height <= rack["height"] <= element_constraints.max_rack_height):
                self.errors.append(f"Rack {i}: Height must be between {element_constraints.min_rack_height} and {element_constraints.max_rack_height}")
    
    def _validate_aisles(self, aisles: List[Dict[str, Any]]) -> None:
        """Validate aisle specifications."""
        element_constraints = ElementConstraints()
        
        for i, aisle in enumerate(aisles):
            if "type" not in aisle:
                self.errors.append(f"Aisle {i}: Missing 'type' field")
                
            if "width" not in aisle:
                self.errors.append(f"Aisle {i}: Missing 'width' field")
            elif not (element_constraints.min_aisle_width <= aisle["width"] <= element_constraints.max_aisle_width):
                self.errors.append(f"Aisle {i}: Width must be between {element_constraints.min_aisle_width} and {element_constraints.max_aisle_width}")
                
            if "direction" not in aisle:
                self.errors.append(f"Aisle {i}: Missing 'direction' field")
            elif aisle["direction"] not in ["horizontal", "vertical"]:
                self.errors.append(f"Aisle {i}: Direction must be 'horizontal' or 'vertical'")
    
    def _validate_zones(self, zones: List[Dict[str, Any]]) -> None:
        """Validate zone specifications."""
        for i, zone in enumerate(zones):
            required_zone_fields = ["name", "type", "priority", "min_area"]
            for field in required_zone_fields:
                if field not in zone:
                    self.errors.append(f"Zone {i}: Missing required field: {field}")
            
            if "priority" in zone and not isinstance(zone["priority"], int):
                self.errors.append(f"Zone {i}: Priority must be an integer")
                
            if "min_area" in zone and zone["min_area"] <= 0:
                self.errors.append(f"Zone {i}: Minimum area must be positive")
                
            if "max_area" in zone and zone["max_area"] <= 0:
                self.errors.append(f"Zone {i}: Maximum area must be positive")
                
            if "min_area" in zone and "max_area" in zone:
                if zone["min_area"] > zone["max_area"]:
                    self.errors.append(f"Zone {i}: Minimum area cannot be greater than maximum area")


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a JSON file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dict containing the configuration
    """
    with open(config_path, 'r') as f:
        return json.load(f)


def validate_config_file(config_path: str) -> bool:
    """
    Validate a configuration file against the schema.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        bool: True if configuration is valid, False otherwise
    """
    try:
        config = load_config(config_path)
        validator = ConfigValidator()
        is_valid = validator.validate_config(config)
        if not is_valid:
            for error in validator.errors:
                print(f"Validation error: {error}")
        return is_valid
    except Exception as e:
        print(f"Failed to validate config file: {str(e)}")
        return False


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
        is_valid = validate_config_file(config_file)
        if is_valid:
            print(f"Configuration file '{config_file}' is valid.")
        else:
            print(f"Configuration file '{config_file}' is invalid.")
            sys.exit(1)
    else:
        print("Usage: python config_schema.py <config_file>")
        sys.exit(1)