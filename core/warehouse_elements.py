"""
Module containing the fundamental elements that make up a warehouse layout.
Each class represents a different warehouse component with its properties and behaviors.
"""
from typing import List, Tuple, Dict, Optional, Any
import uuid
import math
from enum import Enum


class ElementType(Enum):
    """Enumeration of possible warehouse element types."""
    RACK = "rack"
    AISLE = "aisle"
    STAGING_AREA = "staging_area"
    SHIPPING_DOCK = "shipping_dock"
    RECEIVING_DOCK = "receiving_dock"
    OFFICE = "office"
    CHARGING_STATION = "charging_station"
    PACKAGING_AREA = "packaging_area"
    STORAGE_AREA = "storage_area"
    WALL = "wall"
    DOOR = "door"
    COLUMN = "column"
    FIRE_EXIT = "fire_exit"
    CUSTOM = "custom"


class Orientation(Enum):
    """Enumeration of possible orientations for warehouse elements."""
    NORTH = 0
    EAST = 90
    SOUTH = 180
    WEST = 270


class Point:
    """Represents a 2D point in the warehouse layout."""
    
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
    
    def __eq__(self, other):
        if not isinstance(other, Point):
            return False
        return math.isclose(self.x, other.x) and math.isclose(self.y, other.y)
    
    def __str__(self):
        return f"Point({self.x:.2f}, {self.y:.2f})"
    
    def distance_to(self, other: 'Point') -> float:
        """Calculate Euclidean distance to another point."""
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)
    
    def to_tuple(self) -> Tuple[float, float]:
        """Convert to tuple representation."""
        return (self.x, self.y)


class BoundingBox:
    """Represents a rectangular bounding box for warehouse elements."""
    
    def __init__(self, min_x: float, min_y: float, max_x: float, max_y: float):
        self.min_x = min_x
        self.min_y = min_y
        self.max_x = max_x
        self.max_y = max_y
    
    @property
    def width(self) -> float:
        """Width of the bounding box."""
        return self.max_x - self.min_x
    
    @property
    def height(self) -> float:
        """Height of the bounding box."""
        return self.max_y - self.min_y
    
    @property
    def area(self) -> float:
        """Area of the bounding box."""
        return self.width * self.height
    
    @property
    def center(self) -> Point:
        """Center point of the bounding box."""
        return Point(
            (self.min_x + self.max_x) / 2,
            (self.min_y + self.max_y) / 2
        )
    
    def contains_point(self, point: Point) -> bool:
        """Check if the bounding box contains a point."""
        return (self.min_x <= point.x <= self.max_x and 
                self.min_y <= point.y <= self.max_y)
    
    def intersects(self, other: 'BoundingBox') -> bool:
        """Check if this bounding box intersects with another."""
        return not (self.max_x < other.min_x or self.min_x > other.max_x or
                    self.max_y < other.min_y or self.min_y > other.max_y)
    
    def expand(self, margin: float) -> 'BoundingBox':
        """Return a new bounding box expanded by the specified margin."""
        return BoundingBox(
            self.min_x - margin,
            self.min_y - margin,
            self.max_x + margin,
            self.max_y + margin
        )


class WarehouseElement:
    """Base class for all warehouse elements."""
    
    def __init__(self, 
                 element_type: ElementType,
                 position: Point,
                 width: float,
                 length: float,
                 height: Optional[float] = 0.0,
                 orientation: Orientation = Orientation.NORTH,
                 properties: Optional[Dict[str, Any]] = None):
        self.id = str(uuid.uuid4())
        self.element_type = element_type
        self.position = position
        self.width = width
        self.length = length
        self.height = height
        self.orientation = orientation
        self.properties = properties or {}
    
    @property
    def bounding_box(self) -> BoundingBox:
        """Calculate the bounding box based on position, dimensions and orientation."""
        half_width = self.width / 2
        half_length = self.length / 2
        
        # Adjust for orientation
        if self.orientation in [Orientation.NORTH, Orientation.SOUTH]:
            return BoundingBox(
                self.position.x - half_width,
                self.position.y - half_length,
                self.position.x + half_width,
                self.position.y + half_length
            )
        else:  # EAST or WEST
            return BoundingBox(
                self.position.x - half_length,
                self.position.y - half_width,
                self.position.x + half_length,
                self.position.y + half_width
            )
    
    def intersects(self, other: 'WarehouseElement') -> bool:
        """Check if this element intersects with another element."""
        return self.bounding_box.intersects(other.bounding_box)
    
    def distance_to(self, other: 'WarehouseElement') -> float:
        """Calculate distance between the centers of two elements."""
        return self.position.distance_to(other.position)
    
    def rotate(self, degrees: float) -> None:
        """Rotate the element by the specified degrees."""
        new_angle = (self.orientation.value + degrees) % 360
        # Find the closest orientation
        closest_orientation = min(
            Orientation, 
            key=lambda o: abs((o.value - new_angle) % 360)
        )
        self.orientation = closest_orientation
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the element to a dictionary representation."""
        return {
            "id": self.id,
            "type": self.element_type.value,
            "position": {
                "x": self.position.x,
                "y": self.position.y
            },
            "dimensions": {
                "width": self.width,
                "length": self.length,
                "height": self.height
            },
            "orientation": self.orientation.value,
            "properties": self.properties
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WarehouseElement':
        """Create a warehouse element from a dictionary representation."""
        element = cls(
            element_type=ElementType(data["type"]),
            position=Point(data["position"]["x"], data["position"]["y"]),
            width=data["dimensions"]["width"],
            length=data["dimensions"]["length"],
            height=data["dimensions"].get("height", 0.0),
            orientation=Orientation(data["orientation"]),
            properties=data.get("properties", {})
        )
        element.id = data["id"]
        return element


class Rack(WarehouseElement):
    """Represents a storage rack in the warehouse."""
    
    def __init__(self, 
                 position: Point,
                 width: float,
                 length: float,
                 height: float,
                 orientation: Orientation = Orientation.NORTH,
                 num_levels: int = 3,
                 capacity_per_level: float = 1000.0,
                 rack_type: str = "standard",
                 properties: Optional[Dict[str, Any]] = None):
        super().__init__(
            element_type=ElementType.RACK,
            position=position,
            width=width,
            length=length,
            height=height,
            orientation=orientation,
            properties=properties or {}
        )
        self.num_levels = num_levels
        self.capacity_per_level = capacity_per_level
        self.rack_type = rack_type
        
        # Add these to properties as well for serialization
        self.properties.update({
            "num_levels": num_levels,
            "capacity_per_level": capacity_per_level,
            "rack_type": rack_type
        })
    
    @property
    def total_capacity(self) -> float:
        """Calculate the total storage capacity of the rack."""
        return self.num_levels * self.capacity_per_level
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Rack':
        """Create a rack from a dictionary representation."""
        props = data.get("properties", {})
        rack = cls(
            position=Point(data["position"]["x"], data["position"]["y"]),
            width=data["dimensions"]["width"],
            length=data["dimensions"]["length"],
            height=data["dimensions"].get("height", 0.0),
            orientation=Orientation(data["orientation"]),
            num_levels=props.get("num_levels", 3),
            capacity_per_level=props.get("capacity_per_level", 1000.0),
            rack_type=props.get("rack_type", "standard"),
            properties=props
        )
        rack.id = data["id"]
        return rack


class Aisle(WarehouseElement):
    """Represents an aisle in the warehouse."""
    
    def __init__(self, 
                 position: Point,
                 width: float,
                 length: float,
                 orientation: Orientation = Orientation.NORTH,
                 aisle_type: str = "standard",
                 properties: Optional[Dict[str, Any]] = None):
        super().__init__(
            element_type=ElementType.AISLE,
            position=position,
            width=width,
            length=length,
            height=0.0,  # Aisles don't have height
            orientation=orientation,
            properties=properties or {}
        )
        self.aisle_type = aisle_type
        self.properties["aisle_type"] = aisle_type
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Aisle':
        """Create an aisle from a dictionary representation."""
        props = data.get("properties", {})
        aisle = cls(
            position=Point(data["position"]["x"], data["position"]["y"]),
            width=data["dimensions"]["width"],
            length=data["dimensions"]["length"],
            orientation=Orientation(data["orientation"]),
            aisle_type=props.get("aisle_type", "standard"),
            properties=props
        )
        aisle.id = data["id"]
        return aisle


class StationaryElement(WarehouseElement):
    """Base class for stationary elements like docks, offices, etc."""
    
    def __init__(self, 
                 element_type: ElementType,
                 position: Point,
                 width: float,
                 length: float,
                 height: Optional[float] = 0.0,
                 orientation: Orientation = Orientation.NORTH,
                 properties: Optional[Dict[str, Any]] = None):
        super().__init__(
            element_type=element_type,
            position=position,
            width=width,
            length=length,
            height=height,
            orientation=orientation,
            properties=properties or {}
        )


class ShippingDock(StationaryElement):
    """Represents a shipping dock in the warehouse."""
    
    def __init__(self, 
                 position: Point,
                 width: float,
                 length: float,
                 height: Optional[float] = 0.0,
                 orientation: Orientation = Orientation.NORTH,
                 capacity: int = 1,
                 properties: Optional[Dict[str, Any]] = None):
        super().__init__(
            element_type=ElementType.SHIPPING_DOCK,
            position=position,
            width=width,
            length=length,
            height=height,
            orientation=orientation,
            properties=properties or {}
        )
        self.capacity = capacity
        self.properties["capacity"] = capacity


class Office(StationaryElement):
    """Represents an office space in the warehouse."""
    
    def __init__(self, 
                 position: Point,
                 width: float,
                 length: float,
                 height: Optional[float] = 2.5,
                 orientation: Orientation = Orientation.NORTH,
                 num_workstations: int = 1,
                 properties: Optional[Dict[str, Any]] = None):
        super().__init__(
            element_type=ElementType.OFFICE,
            position=position,
            width=width,
            length=length,
            height=height,
            orientation=orientation,
            properties=properties or {}
        )
        self.num_workstations = num_workstations
        self.properties["num_workstations"] = num_workstations


class StagingArea(StationaryElement):
    """Represents a staging area in the warehouse."""
    
    def __init__(self, 
                 position: Point,
                 width: float,
                 length: float,
                 height: Optional[float] = 0.0,
                 orientation: Orientation = Orientation.NORTH,
                 purpose: str = "general",
                 properties: Optional[Dict[str, Any]] = None):
        super().__init__(
            element_type=ElementType.STAGING_AREA,
            position=position,
            width=width,
            length=length,
            height=height,
            orientation=orientation,
            properties=properties or {}
        )
        self.purpose = purpose
        self.properties["purpose"] = purpose


class Column(WarehouseElement):
    """Represents a structural column in the warehouse."""
    
    def __init__(self, 
                 position: Point,
                 width: float,
                 length: Optional[float] = None,
                 height: Optional[float] = 0.0,
                 properties: Optional[Dict[str, Any]] = None):
        # For columns, length=width for square columns
        if length is None:
            length = width
            
        super().__init__(
            element_type=ElementType.COLUMN,
            position=position,
            width=width,
            length=length,
            height=height,
            orientation=Orientation.NORTH,  # Orientation doesn't matter for columns
            properties=properties or {}
        )


class Door(WarehouseElement):
    """Represents a door in the warehouse."""
    
    def __init__(self, 
                 position: Point,
                 width: float,
                 orientation: Orientation = Orientation.NORTH,
                 door_type: str = "standard",
                 properties: Optional[Dict[str, Any]] = None):
        super().__init__(
            element_type=ElementType.DOOR,
            position=position,
            width=width,
            length=0.2,  # Doors are thin
            height=2.0,  # Standard door height
            orientation=orientation,
            properties=properties or {}
        )
        self.door_type = door_type
        self.properties["door_type"] = door_type


class Wall(WarehouseElement):
    """Represents a wall in the warehouse."""
    
    def __init__(self, 
                 start_point: Point,
                 end_point: Point,
                 thickness: float = 0.2,
                 height: Optional[float] = 3.0,
                 properties: Optional[Dict[str, Any]] = None):
        # Calculate center point, length and orientation
        center_x = (start_point.x + end_point.x) / 2
        center_y = (start_point.y + end_point.y) / 2
        length = start_point.distance_to(end_point)
        
        # Calculate angle for orientation
        dx = end_point.x - start_point.x
        dy = end_point.y - start_point.y
        angle = math.degrees(math.atan2(dy, dx))
        
        # Find closest orientation
        if -45 <= angle < 45:
            orientation = Orientation.EAST
        elif 45 <= angle < 135:
            orientation = Orientation.NORTH
        elif -135 <= angle < -45:
            orientation = Orientation.SOUTH
        else:  # angle >= 135 or angle < -135
            orientation = Orientation.WEST
        
        super().__init__(
            element_type=ElementType.WALL,
            position=Point(center_x, center_y),
            width=thickness,
            length=length,
            height=height,
            orientation=orientation,
            properties=properties or {}
        )
        
        # Store the actual endpoints
        self.start_point = start_point
        self.end_point = end_point
        self.properties.update({
            "start_point": {"x": start_point.x, "y": start_point.y},
            "end_point": {"x": end_point.x, "y": end_point.y},
            "thickness": thickness
        })

# Add these classes after the existing classes in warehouse_elements.py

class Shelf(WarehouseElement):
    """Represents a shelf within a storage rack."""
    
    def __init__(self, 
                 position: Point,
                 width: float,
                 length: float,
                 height: float = 0.4,
                 level: int = 0,
                 parent_rack_id: str = None,
                 capacity: float = 100.0,
                 properties: Optional[Dict[str, Any]] = None):
        super().__init__(
            element_type=ElementType.STORAGE_AREA,  # We can add SHELF to ElementType if needed
            position=position,
            width=width,
            length=length,
            height=height,
            orientation=Orientation.NORTH,
            properties=properties or {}
        )
        self.level = level
        self.parent_rack_id = parent_rack_id
        self.capacity = capacity
        
        # Add to properties for serialization
        self.properties.update({
            "level": level,
            "parent_rack_id": parent_rack_id,
            "capacity": capacity
        })
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Shelf':
        """Create a shelf from a dictionary representation."""
        props = data.get("properties", {})
        shelf = cls(
            position=Point(data["position"]["x"], data["position"]["y"]),
            width=data["dimensions"]["width"],
            length=data["dimensions"]["length"],
            height=data["dimensions"].get("height", 0.4),
            level=props.get("level", 0),
            parent_rack_id=props.get("parent_rack_id"),
            capacity=props.get("capacity", 100.0),
            properties=props
        )
        shelf.id = data["id"]
        return shelf


class PickingStation(StationaryElement):
    """Represents a picking station in the warehouse."""
    
    def __init__(self, 
                 position: Point,
                 width: float,
                 length: float,
                 height: Optional[float] = 1.5,
                 orientation: Orientation = Orientation.NORTH,
                 station_type: str = "standard",
                 capacity: int = 5,
                 properties: Optional[Dict[str, Any]] = None):
        super().__init__(
            element_type=ElementType.PICKING_STATION,  # Add to ElementType enum
            position=position,
            width=width,
            length=length,
            height=height,
            orientation=orientation,
            properties=properties or {}
        )
        self.station_type = station_type
        self.capacity = capacity
        
        self.properties.update({
            "station_type": station_type,
            "capacity": capacity
        })
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PickingStation':
        """Create a picking station from a dictionary representation."""
        props = data.get("properties", {})
        station = cls(
            position=Point(data["position"]["x"], data["position"]["y"]),
            width=data["dimensions"]["width"],
            length=data["dimensions"]["length"],
            height=data["dimensions"].get("height", 1.5),
            orientation=Orientation(data["orientation"]),
            station_type=props.get("station_type", "standard"),
            capacity=props.get("capacity", 5),
            properties=props
        )
        station.id = data["id"]
        return station


class StorageZone(WarehouseElement):
    """Represents a storage zone in the warehouse."""
    
    def __init__(self, 
                 position: Point,
                 width: float,
                 length: float,
                 height: Optional[float] = 0.0,
                 orientation: Orientation = Orientation.NORTH,
                 zone_type: str = "general",
                 max_capacity: float = 10000.0,
                 properties: Optional[Dict[str, Any]] = None):
        super().__init__(
            element_type=ElementType.STORAGE_AREA,
            position=position,
            width=width,
            length=length,
            height=height,
            orientation=orientation,
            properties=properties or {}
        )
        self.zone_type = zone_type
        self.max_capacity = max_capacity
        
        self.properties.update({
            "zone_type": zone_type,
            "max_capacity": max_capacity
        })
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StorageZone':
        """Create a storage zone from a dictionary representation."""
        props = data.get("properties", {})
        zone = cls(
            position=Point(data["position"]["x"], data["position"]["y"]),
            width=data["dimensions"]["width"],
            length=data["dimensions"]["length"],
            height=data["dimensions"].get("height", 0.0),
            orientation=Orientation(data["orientation"]),
            zone_type=props.get("zone_type", "general"),
            max_capacity=props.get("max_capacity", 10000.0),
            properties=props
        )
        zone.id = data["id"]
        return zone


class ReceivingArea(StationaryElement):
    """Represents a receiving area in the warehouse."""
    
    def __init__(self, 
                 position: Point,
                 width: float,
                 length: float,
                 height: Optional[float] = 0.0,
                 orientation: Orientation = Orientation.NORTH,
                 processing_capacity: int = 10,
                 properties: Optional[Dict[str, Any]] = None):
        super().__init__(
            element_type=ElementType.RECEIVING_DOCK,
            position=position,
            width=width,
            length=length,
            height=height,
            orientation=orientation,
            properties=properties or {}
        )
        self.processing_capacity = processing_capacity
        
        self.properties.update({
            "processing_capacity": processing_capacity
        })
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ReceivingArea':
        """Create a receiving area from a dictionary representation."""
        props = data.get("properties", {})
        area = cls(
            position=Point(data["position"]["x"], data["position"]["y"]),
            width=data["dimensions"]["width"],
            length=data["dimensions"]["length"],
            height=data["dimensions"].get("height", 0.0),
            orientation=Orientation(data["orientation"]),
            processing_capacity=props.get("processing_capacity", 10),
            properties=props
        )
        area.id = data["id"]
        return area


class ShippingArea(StationaryElement):
    """Represents a shipping area in the warehouse."""
    
    def __init__(self, 
                 position: Point,
                 width: float,
                 length: float,
                 height: Optional[float] = 0.0,
                 orientation: Orientation = Orientation.NORTH,
                 processing_capacity: int = 10,
                 properties: Optional[Dict[str, Any]] = None):
        super().__init__(
            element_type=ElementType.SHIPPING_DOCK,
            position=position,
            width=width,
            length=length,
            height=height,
            orientation=orientation,
            properties=properties or {}
        )
        self.processing_capacity = processing_capacity
        
        self.properties.update({
            "processing_capacity": processing_capacity
        })
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ShippingArea':
        """Create a shipping area from a dictionary representation."""
        props = data.get("properties", {})
        area = cls(
            position=Point(data["position"]["x"], data["position"]["y"]),
            width=data["dimensions"]["width"],
            length=data["dimensions"]["length"],
            height=data["dimensions"].get("height", 0.0),
            orientation=Orientation(data["orientation"]),
            processing_capacity=props.get("processing_capacity", 10),
            properties=props
        )
        area.id = data["id"]
        return area


class RestArea(StationaryElement):
    """Represents a rest area or break room in the warehouse."""
    
    def __init__(self, 
                 position: Point,
                 width: float,
                 length: float,
                 height: Optional[float] = 2.5,
                 orientation: Orientation = Orientation.NORTH,
                 capacity: int = 10,
                 amenities: List[str] = None,
                 properties: Optional[Dict[str, Any]] = None):
        super().__init__(
            element_type=ElementType.CUSTOM,  # Add REST_AREA to ElementType if needed
            position=position,
            width=width,
            length=length,
            height=height,
            orientation=orientation,
            properties=properties or {}
        )
        self.capacity = capacity
        self.amenities = amenities or ["seating"]
        
        self.properties.update({
            "capacity": capacity,
            "amenities": self.amenities
        })
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RestArea':
        """Create a rest area from a dictionary representation."""
        props = data.get("properties", {})
        area = cls(
            position=Point(data["position"]["x"], data["position"]["y"]),
            width=data["dimensions"]["width"],
            length=data["dimensions"]["length"],
            height=data["dimensions"].get("height", 2.5),
            orientation=Orientation(data["orientation"]),
            capacity=props.get("capacity", 10),
            amenities=props.get("amenities", ["seating"]),
            properties=props
        )
        area.id = data["id"]
        return area


# Update ElementType enum by adding these lines in the class:
class ElementType(Enum):
    """Enumeration of possible warehouse element types."""
    # ... (existing types)
    PICKING_STATION = "picking_station"
    SHELF = "shelf"
    REST_AREA = "rest_area"
class WarehouseLayout:
    """Represents the complete layout of a warehouse with all its elements."""
    
    def __init__(self, 
                 width: float, 
                 length: float,
                 name: str = "New Warehouse",
                 description: str = ""):
        self.width = width
        self.length = length
        self.name = name
        self.description = description
        self.elements: List[WarehouseElement] = []
        self.metadata: Dict[str, Any] = {
            "created_at": None,
            "modified_at": None,
            "version": "1.0.0"
        }
    
    def add_element(self, element: WarehouseElement) -> bool:
        """
        Add an element to the layout if it fits within the warehouse bounds.
        Returns True if the element was added, False otherwise.
        """
        # Check if the element fits within the warehouse bounds
        bbox = element.bounding_box
        if (bbox.min_x < 0 or bbox.max_x > self.width or
            bbox.min_y < 0 or bbox.max_y > self.length):
            return False
        
        self.elements.append(element)
        return True
    
    def remove_element(self, element_id: str) -> bool:
        """Remove an element from the layout by its ID."""
        for i, element in enumerate(self.elements):
            if element.id == element_id:
                self.elements.pop(i)
                return True
        return False
    
    def get_element_by_id(self, element_id: str) -> Optional[WarehouseElement]:
        """Get an element by its ID."""
        for element in self.elements:
            if element.id == element_id:
                return element
        return None
    
    def get_elements_by_type(self, element_type: ElementType) -> List[WarehouseElement]:
        """Get all elements of a specific type."""
        return [e for e in self.elements if e.element_type == element_type]
    
    def check_collision(self, element: WarehouseElement) -> bool:
        """Check if an element collides with any existing elements."""
        for existing in self.elements:
            if element.intersects(existing):
                return True
        return False
    
    def get_elements_in_area(self, area: BoundingBox) -> List[WarehouseElement]:
        """Get all elements that intersect with the specified area."""
        return [e for e in self.elements if e.bounding_box.intersects(area)]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the layout to a dictionary representation."""
        return {
            "name": self.name,
            "description": self.description,
            "dimensions": {
                "width": self.width,
                "length": self.length
            },
            "elements": [e.to_dict() for e in self.elements],
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WarehouseLayout':
        """Create a warehouse layout from a dictionary representation."""
        layout = cls(
            width=data["dimensions"]["width"],
            length=data["dimensions"]["length"],
            name=data.get("name", "Imported Warehouse"),
            description=data.get("description", "")
        )
        
        # Add metadata if present
        if "metadata" in data:
            layout.metadata = data["metadata"]
        
        # Add all elements
        for elem_data in data.get("elements", []):
            element_type = ElementType(elem_data["type"])
            
            # Create the appropriate element type
            if element_type == ElementType.RACK:
                element = Rack.from_dict(elem_data)
            elif element_type == ElementType.AISLE:
                element = Aisle.from_dict(elem_data)
            else:
                # General case for other element types
                element = WarehouseElement.from_dict(elem_data)
            
            layout.elements.append(element)
        
        return layout
    
    def calculate_storage_capacity(self) -> float:
        """Calculate the total storage capacity of all racks."""
        total = 0.0
        for element in self.elements:
            if element.element_type == ElementType.RACK and isinstance(element, Rack):
                total += element.total_capacity
        return total
    
    def calculate_floor_area(self) -> float:
        """Calculate the total floor area of the warehouse."""
        return self.width * self.length
    
    def calculate_used_area(self) -> float:
        """Calculate the total area used by all elements."""
        return sum(e.bounding_box.area for e in self.elements)
    
    def calculate_efficiency(self) -> float:
        """Calculate the space efficiency (ratio of used area to total area)."""
        total_area = self.calculate_floor_area()
        if total_area <= 0:
            return 0.0
        return self.calculate_used_area() / total_area
