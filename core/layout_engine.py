"""
Layout Engine - Core component of the Warehouse Layout Generator

This module is responsible for creating and manipulating warehouse layouts.
It handles the spatial arrangement of warehouse elements (racks, aisles, etc.)
according to constraints and optimization criteria.
"""

import logging
import json
from typing import Dict, List, Tuple, Optional, Set, Any, Union
import numpy as np
from dataclasses import dataclass, field

from core.warehouse_elements import (
    WarehouseElement, 
    Rack, 
    Aisle, 
    ReceivingArea, 
    ShippingArea, 
    Wall, 
    Column
)
from core.constraints import ConstraintManager
from utils.geometry import Point, Rectangle, overlap_detection, path_finding
from utils.validators import validate_layout, validate_element_placement
from config.config_schema import LayoutConfig

logger = logging.getLogger(__name__)


@dataclass
class LayoutGrid:
    """A grid representation of the warehouse layout for spatial analysis."""
    width: int
    length: int
    cell_size: float = 0.1  # Default cell size in meters
    grid: np.ndarray = field(init=False)
    
    def __post_init__(self):
        # 0 = free space, 1 = occupied, 2 = reserved for aisles
        self.grid = np.zeros((int(self.length / self.cell_size), 
                               int(self.width / self.cell_size)), 
                             dtype=int)
    
    def mark_occupied(self, rect: Rectangle, value: int = 1) -> None:
        """Mark an area as occupied/reserved on the grid."""
        x1, y1 = int(rect.x / self.cell_size), int(rect.y / self.cell_size)
        x2, y2 = int((rect.x + rect.width) / self.cell_size), int((rect.y + rect.length) / self.cell_size)
        
        # Ensure coordinates are within grid boundaries
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(self.grid.shape[1], x2), min(self.grid.shape[0], y2)
        
        self.grid[y1:y2, x1:x2] = value
    
    def is_area_free(self, rect: Rectangle) -> bool:
        """Check if an area is free (not occupied)."""
        x1, y1 = int(rect.x / self.cell_size), int(rect.y / self.cell_size)
        x2, y2 = int((rect.x + rect.width) / self.cell_size), int((rect.y + rect.length) / self.cell_size)
        
        # Ensure coordinates are within grid boundaries
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(self.grid.shape[1], x2), min(self.grid.shape[0], y2)
        
        return np.all(self.grid[y1:y2, x1:x2] == 0)


class LayoutEngine:
    """
    Main engine for warehouse layout generation and manipulation
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the layout engine with configuration.
        
        Args:
            config: Configuration dictionary for the layout engine
        """
        self.config = config or {}
        self.elements: Dict[str, WarehouseElement] = {}
        self.width = self.config.get('warehouse_width', 50.0)  # Default width in meters
        self.length = self.config.get('warehouse_length', 100.0)  # Default length in meters
        self.grid = LayoutGrid(self.width, self.length)
        self.constraint_manager = ConstraintManager()
        self.element_id_counter = 0
        
        # Initialize with walls if dimensions are provided
        if self.width and self.length:
            self._initialize_warehouse_walls()
            
        logger.info(f"LayoutEngine initialized with dimensions {self.width}m x {self.length}m")
    
    def _initialize_warehouse_walls(self) -> None:
        """Create the outer walls of the warehouse."""
        wall_thickness = self.config.get('wall_thickness', 0.3)  # Default 30cm
        
        # North wall (top)
        self.add_element(Wall(
            x=0, 
            y=0, 
            width=self.width, 
            length=wall_thickness
        ))
        
        # East wall (right)
        self.add_element(Wall(
            x=self.width - wall_thickness, 
            y=0, 
            width=wall_thickness, 
            length=self.length
        ))
        
        # South wall (bottom)
        self.add_element(Wall(
            x=0, 
            y=self.length - wall_thickness, 
            width=self.width, 
            length=wall_thickness
        ))
        
        # West wall (left)
        self.add_element(Wall(
            x=0, 
            y=0, 
            width=wall_thickness, 
            length=self.length
        ))
    
    def _generate_element_id(self, prefix: str) -> str:
        """Generate a unique ID for a warehouse element."""
        self.element_id_counter += 1
        return f"{prefix}_{self.element_id_counter}"
    
    def add_element(self, element: WarehouseElement) -> str:
        """
        Add a warehouse element to the layout.
        
        Args:
            element: The warehouse element to add
            
        Returns:
            The ID of the added element
        
        Raises:
            ValueError: If the element placement violates constraints
        """
        # If the element doesn't have an ID, generate one
        if not hasattr(element, 'id') or not element.id:
            element_type = type(element).__name__.lower()
            element.id = self._generate_element_id(element_type)
        
        # Validate element placement
        validation_result = validate_element_placement(element, self.elements.values())
        if not validation_result['valid']:
            raise ValueError(f"Invalid element placement: {validation_result['reason']}")
        
        # Check constraints
        if not self.constraint_manager.check_constraints(element, self.elements):
            raise ValueError(f"Element placement violates constraints")
        
        # Add element to the layout
        self.elements[element.id] = element
        
        # Update the grid
        rect = Rectangle(element.x, element.y, element.width, element.length)
        self.grid.mark_occupied(rect)
        
        logger.debug(f"Added element {element.id} of type {type(element).__name__}")
        return element.id
    
    def remove_element(self, element_id: str) -> bool:
        """
        Remove a warehouse element from the layout.
        
        Args:
            element_id: The ID of the element to remove
            
        Returns:
            True if the element was removed, False otherwise
        """
        if element_id in self.elements:
            element = self.elements[element_id]
            
            # Update the grid - mark the space as free
            rect = Rectangle(element.x, element.y, element.width, element.length)
            self.grid.mark_occupied(rect, value=0)
            
            # Remove the element
            del self.elements[element_id]
            logger.debug(f"Removed element {element_id}")
            return True
        
        logger.warning(f"Attempted to remove non-existent element {element_id}")
        return False
    
    def move_element(self, element_id: str, new_x: float, new_y: float) -> bool:
        """
        Move a warehouse element to a new position.
        
        Args:
            element_id: The ID of the element to move
            new_x: The new x-coordinate
            new_y: The new y-coordinate
            
        Returns:
            True if the element was moved, False otherwise
        """
        if element_id not in self.elements:
            logger.warning(f"Attempted to move non-existent element {element_id}")
            return False
        
        element = self.elements[element_id]
        
        # Store original position
        original_x, original_y = element.x, element.y
        
        # Temporarily remove the element from the layout
        self.remove_element(element_id)
        
        # Update position
        element.x, element.y = new_x, new_y
        
        try:
            # Try to add the element at the new position
            self.add_element(element)
            return True
        except ValueError as e:
            # If there's an error, restore the element to its original position
            element.x, element.y = original_x, original_y
            self.add_element(element)
            logger.warning(f"Failed to move element {element_id}: {str(e)}")
            return False
    
    def resize_element(self, element_id: str, new_width: float, new_length: float) -> bool:
        """
        Resize a warehouse element.
        
        Args:
            element_id: The ID of the element to resize
            new_width: The new width
            new_length: The new length
            
        Returns:
            True if the element was resized, False otherwise
        """
        if element_id not in self.elements:
            logger.warning(f"Attempted to resize non-existent element {element_id}")
            return False
        
        element = self.elements[element_id]
        
        # Store original dimensions
        original_width, original_length = element.width, element.length
        
        # Temporarily remove the element from the layout
        self.remove_element(element_id)
        
        # Update dimensions
        element.width, element.length = new_width, new_length
        
        try:
            # Try to add the element with the new dimensions
            self.add_element(element)
            return True
        except ValueError as e:
            # If there's an error, restore the element to its original dimensions
            element.width, element.length = original_width, original_length
            self.add_element(element)
            logger.warning(f"Failed to resize element {element_id}: {str(e)}")
            return False
    
    def find_free_space(self, width: float, length: float) -> Optional[Tuple[float, float]]:
        """
        Find a free space in the layout for an element of the given dimensions.
        
        Args:
            width: The width of the element
            length: The length of the element
            
        Returns:
            A tuple of (x, y) coordinates if a space is found, None otherwise
        """
        # Simple implementation - iterates through grid cells
        cell_size = self.grid.cell_size
        grid_width = self.grid.grid.shape[1]
        grid_length = self.grid.grid.shape[0]
        
        # Convert element dimensions to grid cells
        width_cells = int(width / cell_size) + 1
        length_cells = int(length / cell_size) + 1
        
        # Scan the grid for a free space
        for y in range(0, grid_length - length_cells + 1, 1):
            for x in range(0, grid_width - width_cells + 1, 1):
                # Check if this area is free
                if np.all(self.grid.grid[y:y+length_cells, x:x+width_cells] == 0):
                    # Convert grid coordinates back to meters
                    real_x = x * cell_size
                    real_y = y * cell_size
                    return (real_x, real_y)
        
        # No free space found
        return None
    
    def get_element_at_position(self, x: float, y: float) -> Optional[WarehouseElement]:
        """
        Get the warehouse element at a specific position.
        
        Args:
            x: The x-coordinate
            y: The y-coordinate
            
        Returns:
            The warehouse element at the position, or None if no element is found
        """
        point = Point(x, y)
        
        for element in self.elements.values():
            element_rect = Rectangle(element.x, element.y, element.width, element.length)
            if point.is_inside(element_rect):
                return element
        
        return None
    
    def get_elements_by_type(self, element_type: type) -> List[WarehouseElement]:
        """
        Get all warehouse elements of a specific type.
        
        Args:
            element_type: The type of elements to get
            
        Returns:
            A list of warehouse elements of the specified type
        """
        return [element for element in self.elements.values() if isinstance(element, element_type)]
    
    def get_layout_data(self) -> Dict:
        """
        Get the complete layout data as a dictionary.
        
        Returns:
            A dictionary containing all layout data
        """
        layout_data = {
            'dimensions': {
                'width': self.width,
                'length': self.length
            },
            'elements': {}
        }
        
        # Group elements by type
        for element_id, element in self.elements.items():
            element_type = type(element).__name__
            
            if element_type not in layout_data['elements']:
                layout_data['elements'][element_type] = []
            
            # Serialize the element
            element_data = {
                'id': element.id,
                'x': element.x,
                'y': element.y,
                'width': element.width,
                'length': element.length
            }
            
            # Add additional attributes based on element type
            if hasattr(element, 'height'):
                element_data['height'] = element.height
            
            if hasattr(element, 'capacity'):
                element_data['capacity'] = element.capacity
            
            if hasattr(element, 'orientation'):
                element_data['orientation'] = element.orientation
                
            layout_data['elements'][element_type].append(element_data)
        
        return layout_data
    
    def load_layout_from_data(self, layout_data: Dict) -> None:
        """
        Load a layout from a data dictionary.
        
        Args:
            layout_data: The layout data dictionary
        """
        # Clear current layout
        self.elements = {}
        
        # Set dimensions
        if 'dimensions' in layout_data:
            self.width = layout_data['dimensions'].get('width', self.width)
            self.length = layout_data['dimensions'].get('length', self.length)
            
            # Reinitialize grid
            self.grid = LayoutGrid(self.width, self.length)
        
        # Create elements
        if 'elements' in layout_data:
            element_mapping = {
                'Rack': Rack,
                'Aisle': Aisle,
                'ReceivingArea': ReceivingArea,
                'ShippingArea': ShippingArea,
                'Wall': Wall,
                'Column': Column
            }
            
            for element_type, elements in layout_data['elements'].items():
                if element_type in element_mapping:
                    element_class = element_mapping[element_type]
                    
                    for element_data in elements:
                        # Create element instance with basic attributes
                        element = element_class(
                            x=element_data['x'],
                            y=element_data['y'],
                            width=element_data['width'],
                            length=element_data['length']
                        )
                        
                        # Set the ID
                        element.id = element_data.get('id', self._generate_element_id(element_type.lower()))
                        
                        # Set additional attributes if they exist
                        if 'height' in element_data and hasattr(element, 'height'):
                            element.height = element_data['height']
                        
                        if 'capacity' in element_data and hasattr(element, 'capacity'):
                            element.capacity = element_data['capacity']
                        
                        if 'orientation' in element_data and hasattr(element, 'orientation'):
                            element.orientation = element_data['orientation']
                        
                        # Add the element to the layout (bypassing constraint checks)
                        self.elements[element.id] = element
                        
                        # Update the grid
                        rect = Rectangle(element.x, element.y, element.width, element.length)
                        self.grid.mark_occupied(rect)
        
        logger.info(f"Loaded layout with {len(self.elements)} elements")
    
    def save_layout_to_file(self, filename: str) -> None:
        """
        Save the current layout to a JSON file.
        
        Args:
            filename: The name of the file to save to
        """
        layout_data = self.get_layout_data()
        
        with open(filename, 'w') as f:
            json.dump(layout_data, f, indent=4)
        
        logger.info(f"Saved layout to {filename}")
    
    def load_layout_from_file(self, filename: str) -> None:
        """
        Load a layout from a JSON file.
        
        Args:
            filename: The name of the file to load from
        """
        with open(filename, 'r') as f:
            layout_data = json.load(f)
        
        self.load_layout_from_data(layout_data)
        logger.info(f"Loaded layout from {filename}")
    
    def validate_layout(self) -> Dict:
        """
        Validate the current layout based on constraints and rules.
        
        Returns:
            A dictionary with validation results
        """
        return validate_layout(self.elements.values(), self.width, self.length)
    
    def auto_generate_aisles(self, aisle_width: float = 3.0) -> List[str]:
        """
        Automatically generate aisles between racks.
        
        Args:
            aisle_width: The width of the aisles to generate
            
        Returns:
            A list of IDs of the generated aisles
        """
        generated_aisles = []
        racks = self.get_elements_by_type(Rack)
        
        # Simple algorithm to place aisles between racks
        for i, rack1 in enumerate(racks):
            for rack2 in racks[i+1:]:
                # Check if racks are aligned and close enough for an aisle
                if (abs(rack1.x - rack2.x) <= rack1.width + rack2.width + aisle_width * 2 and
                    (rack1.y <= rack2.y <= rack1.y + rack1.length or 
                     rack2.y <= rack1.y <= rack2.y + rack2.length)):
                    
                    # Create an aisle between these racks
                    min_x = min(rack1.x + rack1.width, rack2.x + rack2.width)
                    max_x = max(rack1.x, rack2.x)
                    
                    aisle_x = min_x
                    aisle_width_actual = max_x - min_x
                    
                    # Calculate aisle length based on rack positions
                    min_y = min(rack1.y, rack2.y)
                    max_length = max(rack1.y + rack1.length, rack2.y + rack2.length) - min_y
                    
                    try:
                        aisle = Aisle(
                            x=aisle_x,
                            y=min_y,
                            width=aisle_width_actual,
                            length=max_length
                        )
                        
                        aisle_id = self.add_element(aisle)
                        generated_aisles.append(aisle_id)
                    except ValueError:
                        # Skip if aisle creation fails due to constraints
                        continue
        
        logger.info(f"Auto-generated {len(generated_aisles)} aisles")
        return generated_aisles
    
    def calculate_efficiency_metrics(self) -> Dict:
        """
        Calculate efficiency metrics for the current layout.
        
        Returns:
            A dictionary containing efficiency metrics
        """
        total_area = self.width * self.length
        storage_area = sum(element.width * element.length 
                          for element in self.get_elements_by_type(Rack))
        aisle_area = sum(element.width * element.length 
                        for element in self.get_elements_by_type(Aisle))
        
        # Calculate distance metrics
        receiving_areas = self.get_elements_by_type(ReceivingArea)
        shipping_areas = self.get_elements_by_type(ShippingArea)
        racks = self.get_elements_by_type(Rack)
        
        # Average distance from receiving to storage
        avg_receiving_distance = 0
        if receiving_areas and racks:
            distances = []
            for receiving in receiving_areas:
                receive_point = Point(receiving.x + receiving.width/2, receiving.y + receiving.length/2)
                for rack in racks:
                    rack_point = Point(rack.x + rack.width/2, rack.y + rack.length/2)
                    distances.append(receive_point.distance_to(rack_point))
            
            avg_receiving_distance = sum(distances) / len(distances) if distances else 0
        
        # Average distance from storage to shipping
        avg_shipping_distance = 0
        if shipping_areas and racks:
            distances = []
            for shipping in shipping_areas:
                ship_point = Point(shipping.x + shipping.width/2, shipping.y + shipping.length/2)
                for rack in racks:
                    rack_point = Point(rack.x + rack.width/2, rack.y + rack.length/2)
                    distances.append(ship_point.distance_to(rack_point))
            
            avg_shipping_distance = sum(distances) / len(distances) if distances else 0
        
        return {
            'total_area': total_area,
            'storage_area': storage_area,
            'aisle_area': aisle_area,
            'storage_ratio': storage_area / total_area if total_area else 0,
            'aisle_ratio': aisle_area / total_area if total_area else 0,
            'unused_area': total_area - storage_area - aisle_area,
            'avg_receiving_distance': avg_receiving_distance,
            'avg_shipping_distance': avg_shipping_distance
        }


if __name__ == "__main__":
    # Simple example usage
    layout_engine = LayoutEngine({
        'warehouse_width': 50.0,
        'warehouse_length': 100.0
    })
    
    # Add some racks
    rack1 = Rack(x=5.0, y=5.0, width=2.0, length=10.0, height=3.0)
    rack_id = layout_engine.add_element(rack1)
    
    # Add a receiving area
    receiving = ReceivingArea(x=2.0, y=20.0, width=5.0, length=5.0)
    layout_engine.add_element(receiving)
    
    # Print layout data
    layout_data = layout_engine.get_layout_data()
    print(json.dumps(layout_data, indent=2))