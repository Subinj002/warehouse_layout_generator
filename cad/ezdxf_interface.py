"""
ezdxf_interface.py
Interface for creating and modifying DXF files using the ezdxf library.
This module handles the interaction between the warehouse layout generator
and the DXF file format commonly used in CAD applications.
"""

import os
import ezdxf
from ezdxf.enums import TextEntityAlignment
from typing import Dict, List, Tuple, Optional, Union, Any

from core.warehouse_elements import (
    Rack, Aisle, Wall, Door, Column, 
    LoadingDock, Zone, Equipment, Obstacle
)


class DXFInterface:
    """
    Interface for creating and manipulating DXF files for warehouse layouts.
    """
    
    def __init__(self, units: str = 'METERS'):
        """
        Initialize the DXF interface.
        
        Args:
            units: The units to use for the drawing ('METERS', 'FEET', etc.)
        """
        self.drawing = None
        self.units = units
        self.layers = {}
        self.styles = {}
        
        # Layer colors
        self.layer_colors = {
            'WALL': 1,          # Red
            'RACK': 5,          # Blue
            'AISLE': 3,         # Green
            'COLUMN': 6,        # Magenta
            'DOOR': 2,          # Yellow
            'LOADING_DOCK': 4,  # Cyan
            'ZONE': 30,         # Orange
            'EQUIPMENT': 41,    # Purple
            'OBSTACLE': 21,     # Brown
            'DIMENSIONS': 7,    # White
            'TEXT': 7,          # White
            'TITLE': 7          # White
        }
    
    def create_new_drawing(self, filename: str) -> None:
        """
        Create a new DXF drawing file.
        
        Args:
            filename: Name of the file to create
        """
        self.drawing = ezdxf.new(dxfversion='R2010')
        self._setup_drawing()
        self.save_drawing(filename)
    
    def open_drawing(self, filename: str) -> None:
        """
        Open an existing DXF drawing file.
        
        Args:
            filename: Path to the DXF file to open
        
        Raises:
            FileNotFoundError: If the file doesn't exist
            ezdxf.DXFError: If the file is not a valid DXF file
        """
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File {filename} not found")
        
        try:
            self.drawing = ezdxf.readfile(filename)
            self._update_layers_from_drawing()
        except ezdxf.DXFError as e:
            raise ezdxf.DXFError(f"Error opening DXF file: {str(e)}")
    
    def save_drawing(self, filename: str) -> None:
        """
        Save the current drawing to a DXF file.
        
        Args:
            filename: Path where to save the DXF file
        """
        if self.drawing is None:
            raise ValueError("No drawing to save")
        
        if not filename.endswith('.dxf'):
            filename += '.dxf'
        
        self.drawing.saveas(filename)
    
    def _setup_drawing(self) -> None:
        """Set up the drawing with standard layers and styles."""
        self._create_standard_layers()
        self._create_text_styles()
        self._setup_modelspace()
    
    def _create_standard_layers(self) -> None:
        """Create standard layers for warehouse elements."""
        # Create the layers
        for layer_name, color in self.layer_colors.items():
            self.create_layer(layer_name, color)
    
    def _create_text_styles(self) -> None:
        """Create text styles for the drawing."""
        # Add some standard text styles
        styles = {
            'STANDARD': {'font': 'arial.ttf', 'height': 0.25},
            'TITLE': {'font': 'arial.ttf', 'height': 0.5, 'bold': True},
            'LABEL': {'font': 'arial.ttf', 'height': 0.18}
        }
        
        for style_name, style_props in styles.items():
            self.create_text_style(style_name, **style_props)
    
    def _setup_modelspace(self) -> None:
        """Set up the modelspace with initial settings."""
        pass  # For now, no initial settings needed
    
    def _update_layers_from_drawing(self) -> None:
        """Update the layer dictionary from the loaded drawing."""
        if self.drawing is None:
            return
        
        self.layers = {layer.dxf.name: layer for layer in self.drawing.layers}
    
    def create_layer(self, name: str, color: int = 7, linetype: str = 'CONTINUOUS') -> None:
        """
        Create a new layer in the drawing.
        
        Args:
            name: Name of the layer
            color: Color number (1-255)
            linetype: Line type name
        """
        if self.drawing is None:
            raise ValueError("No drawing available")
        
        # Create the layer if it doesn't exist
        if name not in self.drawing.layers:
            layer = self.drawing.layers.add(name)
            layer.color = color
            layer.linetype = linetype
            self.layers[name] = layer
    
    def create_text_style(self, name: str, font: str = 'arial.ttf', 
                         height: float = 0.25, bold: bool = False) -> None:
        """
        Create a text style for the drawing.
        
        Args:
            name: Name of the style
            font: Font file name
            height: Text height
            bold: Whether text should be bold
        """
        if self.drawing is None:
            raise ValueError("No drawing available")
        
        # Create the text style if it doesn't exist
        if name not in self.drawing.styles:
            style = self.drawing.styles.add(name)
            style.dxf.font = font
            style.dxf.height = height
            # Note: ezdxf doesn't directly support bold property
            # It would need to be specified through the font file
            self.styles[name] = style
    
    def add_rack(self, rack: Rack, layer: str = 'RACK') -> None:
        """
        Add a rack to the drawing.
        
        Args:
            rack: Rack object with position and dimensions
            layer: Layer to add the rack to
        """
        if self.drawing is None:
            raise ValueError("No drawing available")
        
        # Ensure the layer exists
        if layer not in self.drawing.layers:
            self.create_layer(layer, self.layer_colors.get(layer, 7))
        
        msp = self.drawing.modelspace()
        
        # Draw the rack as a rectangle
        x, y = rack.position
        width, depth = rack.dimensions
        height = rack.height
        
        # Create the rack outline
        points = [
            (x, y),                     # Bottom left
            (x + width, y),             # Bottom right
            (x + width, y + depth),     # Top right
            (x, y + depth),             # Top left
            (x, y)                      # Back to start
        ]
        
        msp.add_lwpolyline(points, dxfattribs={'layer': layer})
        
        # Add a label for the rack
        if hasattr(rack, 'id') and rack.id:
            self.add_text(
                f"Rack {rack.id}",
                (x + width/2, y + depth/2),
                layer='TEXT',
                height=0.15,
                alignment=TextEntityAlignment.MIDDLE_CENTER
            )
    
    def add_aisle(self, aisle: Aisle, layer: str = 'AISLE') -> None:
        """
        Add an aisle to the drawing.
        
        Args:
            aisle: Aisle object with start point, end point and width
            layer: Layer to add the aisle to
        """
        if self.drawing is None:
            raise ValueError("No drawing available")
        
        # Ensure the layer exists
        if layer not in self.drawing.layers:
            self.create_layer(layer, self.layer_colors.get(layer, 7))
        
        msp = self.drawing.modelspace()
        
        # Get aisle coordinates
        start_x, start_y = aisle.start_point
        end_x, end_y = aisle.end_point
        width = aisle.width
        
        # Calculate the perpendicular direction to create width
        length = ((end_x - start_x)**2 + (end_y - start_y)**2)**0.5
        if length == 0:
            return  # Can't draw an aisle with zero length
        
        dx = (end_x - start_x) / length
        dy = (end_y - start_y) / length
        
        # Perpendicular direction
        px, py = -dy, dx
        
        # Calculate the four corners of the aisle
        half_width = width / 2
        points = [
            (start_x + px * half_width, start_y + py * half_width),
            (end_x + px * half_width, end_y + py * half_width),
            (end_x - px * half_width, end_y - py * half_width),
            (start_x - px * half_width, start_y - py * half_width),
            (start_x + px * half_width, start_y + py * half_width)  # Close the polygon
        ]
        
        msp.add_lwpolyline(points, dxfattribs={'layer': layer})
        
        # Add a label for the aisle
        if hasattr(aisle, 'id') and aisle.id:
            center_x = (start_x + end_x) / 2
            center_y = (start_y + end_y) / 2
            self.add_text(
                f"Aisle {aisle.id}",
                (center_x, center_y),
                layer='TEXT',
                height=0.15,
                alignment=TextEntityAlignment.MIDDLE_CENTER
            )
    
    def add_wall(self, wall: Wall, layer: str = 'WALL') -> None:
        """
        Add a wall to the drawing.
        
        Args:
            wall: Wall object with start point, end point and thickness
            layer: Layer to add the wall to
        """
        if self.drawing is None:
            raise ValueError("No drawing available")
        
        # Ensure the layer exists
        if layer not in self.drawing.layers:
            self.create_layer(layer, self.layer_colors.get(layer, 7))
        
        msp = self.drawing.modelspace()
        
        # Get wall coordinates
        start_x, start_y = wall.start_point
        end_x, end_y = wall.end_point
        thickness = wall.thickness
        
        # Calculate the perpendicular direction to create thickness
        length = ((end_x - start_x)**2 + (end_y - start_y)**2)**0.5
        if length == 0:
            return  # Can't draw a wall with zero length
        
        dx = (end_x - start_x) / length
        dy = (end_y - start_y) / length
        
        # Perpendicular direction
        px, py = -dy, dx
        
        # Calculate the four corners of the wall
        half_thickness = thickness / 2
        points = [
            (start_x + px * half_thickness, start_y + py * half_thickness),
            (end_x + px * half_thickness, end_y + py * half_thickness),
            (end_x - px * half_thickness, end_y - py * half_thickness),
            (start_x - px * half_thickness, start_y - py * half_thickness),
            (start_x + px * half_thickness, start_y + py * half_thickness)  # Close the polygon
        ]
        
        msp.add_lwpolyline(points, dxfattribs={'layer': layer})
    
    def add_door(self, door: Door, layer: str = 'DOOR') -> None:
        """
        Add a door to the drawing.
        
        Args:
            door: Door object with position, width and orientation
            layer: Layer to add the door to
        """
        if self.drawing is None:
            raise ValueError("No drawing available")
        
        # Ensure the layer exists
        if layer not in self.drawing.layers:
            self.create_layer(layer, self.layer_colors.get(layer, 7))
        
        msp = self.drawing.modelspace()
        
        # Get door coordinates
        x, y = door.position
        width = door.width
        angle = door.orientation  # In degrees
        
        # Create the door symbol (an arc with a line)
        # First the door frame
        start_x, start_y = x, y
        end_x, end_y = x + width * ezdxf.math.cos_deg(angle), y + width * ezdxf.math.sin_deg(angle)
        
        msp.add_line((start_x, start_y), (end_x, end_y), dxfattribs={'layer': layer})
        
        # Now add the door swing arc
        # The radius of the arc is the width of the door
        radius = width
        
        # Draw the arc from the hinge point
        # The arc spans 90 degrees starting from the door orientation
        msp.add_arc(
            center=(start_x, start_y),
            radius=radius,
            start_angle=angle,
            end_angle=angle + 90,
            dxfattribs={'layer': layer}
        )
        
        # Add a label for the door
        if hasattr(door, 'id') and door.id:
            center_x = (start_x + end_x) / 2
            center_y = (start_y + end_y) / 2
            self.add_text(
                f"Door {door.id}",
                (center_x, center_y),
                layer='TEXT',
                height=0.15,
                alignment=TextEntityAlignment.MIDDLE_CENTER
            )
    
    def add_column(self, column: Column, layer: str = 'COLUMN') -> None:
        """
        Add a column to the drawing.
        
        Args:
            column: Column object with position and dimensions
            layer: Layer to add the column to
        """
        if self.drawing is None:
            raise ValueError("No drawing available")
        
        # Ensure the layer exists
        if layer not in self.drawing.layers:
            self.create_layer(layer, self.layer_colors.get(layer, 7))
        
        msp = self.drawing.modelspace()
        
        # Get column coordinates
        x, y = column.position
        width, depth = column.dimensions
        
        # Draw the column as a filled rectangle
        points = [
            (x, y),                     # Bottom left
            (x + width, y),             # Bottom right
            (x + width, y + depth),     # Top right
            (x, y + depth),             # Top left
            (x, y)                      # Back to start
        ]
        
        msp.add_lwpolyline(points, dxfattribs={'layer': layer})
        
        # Add crosshatch to indicate it's a column
        diag1 = [(x, y), (x + width, y + depth)]
        diag2 = [(x + width, y), (x, y + depth)]
        
        msp.add_line(diag1[0], diag1[1], dxfattribs={'layer': layer})
        msp.add_line(diag2[0], diag2[1], dxfattribs={'layer': layer})
    
    def add_loading_dock(self, dock: LoadingDock, layer: str = 'LOADING_DOCK') -> None:
        """
        Add a loading dock to the drawing.
        
        Args:
            dock: LoadingDock object with position and dimensions
            layer: Layer to add the loading dock to
        """
        if self.drawing is None:
            raise ValueError("No drawing available")
        
        # Ensure the layer exists
        if layer not in self.drawing.layers:
            self.create_layer(layer, self.layer_colors.get(layer, 7))
        
        msp = self.drawing.modelspace()
        
        # Get dock coordinates
        x, y = dock.position
        width, depth = dock.dimensions
        
        # Draw the dock as a rectangle with internal details
        # Outer rectangle
        points = [
            (x, y),                     # Bottom left
            (x + width, y),             # Bottom right
            (x + width, y + depth),     # Top right
            (x, y + depth),             # Top left
            (x, y)                      # Back to start
        ]
        
        msp.add_lwpolyline(points, dxfattribs={'layer': layer})
        
        # Add some details to indicate it's a loading dock
        # Add a ramp symbol
        ramp_width = width * 0.8
        ramp_start_x = x + (width - ramp_width) / 2
        ramp_end_x = ramp_start_x + ramp_width
        
        # Draw ramp lines
        num_lines = 3
        for i in range(num_lines):
            y_pos = y + depth * (i + 1) / (num_lines + 1)
            msp.add_line(
                (ramp_start_x, y_pos), 
                (ramp_end_x, y_pos), 
                dxfattribs={'layer': layer}
            )
        
        # Add a label for the dock
        if hasattr(dock, 'id') and dock.id:
            self.add_text(
                f"Dock {dock.id}",
                (x + width/2, y + depth/2),
                layer='TEXT',
                height=0.15,
                alignment=TextEntityAlignment.MIDDLE_CENTER
            )
    
    def add_zone(self, zone: Zone, layer: str = 'ZONE') -> None:
        """
        Add a zone to the drawing.
        
        Args:
            zone: Zone object with boundary points
            layer: Layer to add the zone to
        """
        if self.drawing is None:
            raise ValueError("No drawing available")
        
        # Ensure the layer exists
        if layer not in self.drawing.layers:
            self.create_layer(layer, self.layer_colors.get(layer, 7))
        
        msp = self.drawing.modelspace()
        
        # Get zone boundary points
        boundary = zone.boundary
        if len(boundary) < 3:
            return  # Need at least 3 points for a zone
        
        # Close the polygon if it's not already closed
        if boundary[0] != boundary[-1]:
            boundary.append(boundary[0])
        
        # Draw the zone boundary
        msp.add_lwpolyline(boundary, dxfattribs={'layer': layer})
        
        # Add a label for the zone
        if hasattr(zone, 'id') and zone.id:
            # Calculate the centroid of the zone
            x_sum = sum(p[0] for p in boundary)
            y_sum = sum(p[1] for p in boundary)
            centroid_x = x_sum / len(boundary)
            centroid_y = y_sum / len(boundary)
            
            self.add_text(
                f"Zone {zone.id}: {zone.zone_type}",
                (centroid_x, centroid_y),
                layer='TEXT',
                height=0.2,
                alignment=TextEntityAlignment.MIDDLE_CENTER
            )
    
    def add_equipment(self, equipment: Equipment, layer: str = 'EQUIPMENT') -> None:
        """
        Add equipment to the drawing.
        
        Args:
            equipment: Equipment object with position and dimensions
            layer: Layer to add the equipment to
        """
        if self.drawing is None:
            raise ValueError("No drawing available")
        
        # Ensure the layer exists
        if layer not in self.drawing.layers:
            self.create_layer(layer, self.layer_colors.get(layer, 7))
        
        msp = self.drawing.modelspace()
        
        # Get equipment coordinates
        x, y = equipment.position
        width, depth = equipment.dimensions
        
        # Draw the equipment as a rectangle
        points = [
            (x, y),                     # Bottom left
            (x + width, y),             # Bottom right
            (x + width, y + depth),     # Top right
            (x, y + depth),             # Top left
            (x, y)                      # Back to start
        ]
        
        msp.add_lwpolyline(points, dxfattribs={'layer': layer})
        
        # Add a label for the equipment
        if hasattr(equipment, 'equipment_type') and equipment.equipment_type:
            self.add_text(
                f"{equipment.equipment_type}",
                (x + width/2, y + depth/2),
                layer='TEXT',
                height=0.15,
                alignment=TextEntityAlignment.MIDDLE_CENTER
            )
    
    def add_obstacle(self, obstacle: Obstacle, layer: str = 'OBSTACLE') -> None:
        """
        Add an obstacle to the drawing.
        
        Args:
            obstacle: Obstacle object with boundary points
            layer: Layer to add the obstacle to
        """
        if self.drawing is None:
            raise ValueError("No drawing available")
        
        # Ensure the layer exists
        if layer not in self.drawing.layers:
            self.create_layer(layer, self.layer_colors.get(layer, 7))
        
        msp = self.drawing.modelspace()
        
        # Get obstacle boundary points
        boundary = obstacle.boundary
        if len(boundary) < 3:
            return  # Need at least 3 points for an obstacle
        
        # Close the polygon if it's not already closed
        if boundary[0] != boundary[-1]:
            boundary.append(boundary[0])
        
        # Draw the obstacle boundary
        msp.add_lwpolyline(boundary, dxfattribs={'layer': layer})
        
        # Add crosshatching to indicate it's an obstacle
        # For simplicity, just adding diagonal lines
        if len(boundary) > 3:  # Only add diagonals if we have more than a triangle
            msp.add_line(boundary[0], boundary[2], dxfattribs={'layer': layer})
            msp.add_line(boundary[1], boundary[3], dxfattribs={'layer': layer})
    
    def add_text(self, text: str, position: Tuple[float, float], 
                layer: str = 'TEXT', height: float = 0.25, 
                style: str = 'STANDARD', 
                alignment: TextEntityAlignment = TextEntityAlignment.LEFT) -> None:
        """
        Add text to the drawing.
        
        Args:
            text: Text content
            position: (x, y) coordinates
            layer: Layer to add the text to
            height: Text height
            style: Text style name
            alignment: Text alignment
        """
        if self.drawing is None:
            raise ValueError("No drawing available")
        
        # Ensure the layer exists
        if layer not in self.drawing.layers:
            self.create_layer(layer, self.layer_colors.get(layer, 7))
        
        msp = self.drawing.modelspace()
        
        # Create the text
        msp.add_text(
            text,
            dxfattribs={
                'layer': layer,
                'height': height,
                'style': style if style in self.styles else 'STANDARD'
            }
        ).set_placement(position, align=alignment)
    
    def add_dimension(self, start: Tuple[float, float], end: Tuple[float, float], 
                     layer: str = 'DIMENSIONS', offset: float = 0.5) -> None:
        """
        Add a linear dimension to the drawing.
        
        Args:
            start: Start point (x, y)
            end: End point (x, y)
            layer: Layer to add the dimension to
            offset: Distance of dimension line from the object
        """
        if self.drawing is None:
            raise ValueError("No drawing available")
        
        # Ensure the layer exists
        if layer not in self.drawing.layers:
            self.create_layer(layer, self.layer_colors.get(layer, 7))
        
        msp = self.drawing.modelspace()
        
        # Calculate if this is more horizontal or vertical
        dx = abs(end[0] - start[0])
        dy = abs(end[1] - start[1])
        
        if dx >= dy:  # Horizontal dimension
            # The dimension line will be offset in the y direction
            dim_line_y = start[1] + offset if start[1] <= end[1] else start[1] - offset
            location = (start[0], dim_line_y)
            msp.add_linear_dimension(
                base=(start[0], start[1]),
                other=(end[0], start[1]),
                location=location,
                dxfattribs={'layer': layer}
            )
        else:  # Vertical dimension
            # The dimension line will be offset in the x direction
            dim_line_x = start[0] + offset if start[0] <= end[0] else start[0] - offset
            location = (dim_line_x, start[1])
            msp.add_linear_dimension(
                base=(start[0], start[1]),
                other=(start[0], end[1]),
                location=location,
                dxfattribs={'layer': layer}
            )
    
    def add_title_block(self, title: str, scale: str = "1:100", 
                       drawn_by: str = "", checked_by: str = "",
                       date: str = "", project_no: str = "") -> None:
        """
        Add a title block to the drawing.
        
        Args:
            title: Drawing title
            scale: Drawing scale
            drawn_by: Name of the person who drew the layout
            checked_by: Name of the person who checked the layout
            date: Date of drawing
            project_no: Project number
        """
        if self.drawing is None:
            raise ValueError("No drawing available")
        
        msp = self.drawing.modelspace()
        
        # Create a title block layer if it doesn't exist
        if 'TITLE' not in self.drawing.layers:
            self.create_layer('TITLE', self.layer_colors.get('TITLE', 7))
        
        # Add a title block at the bottom right of the drawing
        # For now, just add the title text
        # In a real application, you'd add a proper title block with a box and fields
        self.add_text(
            f"TITLE: {title}",
            (0, -2),
            layer='TITLE',
            height=0.3,
            style='TITLE',
            alignment=TextEntityAlignment.LEFT
        )
        
        self.add_text(
            f"SCALE: {scale}",
            (0, -2.5),
            layer='TITLE',
            height=0.2,
            style='STANDARD',
            alignment=TextEntityAlignment.LEFT
        )
        
        if drawn_by:
            self.add_text(
                f"DRAWN BY: {drawn_by}",
                (0, -3),
                layer='TITLE',
                height=0.2,
                style='STANDARD',
                alignment=TextEntityAlignment.LEFT
            )
        
        if checked_by:
            self.add_text(
                f"CHECKED BY: {checked_by}",
                (0, -3.5),
                layer='TITLE',
                height=0.2,
                style='STANDARD',
                alignment=TextEntityAlignment.LEFT
            )
        
        if date:
            self.add_text(
                f"DATE: {date}",
                (0, -4),
                layer='TITLE',
                height=0.2,
                style='STANDARD',
                alignment=TextEntityAlignment.LEFT
            )
        
        if project_no:
            self.add_text(
                f"PROJECT NO: {project_no}",
                (0, -4.5),
                layer='TITLE',
                height=0.2,
                style='STANDARD',
                alignment=TextEntityAlignment.LEFT
            )
    
    def set_units(self, units: str) -> None:
        """
        Set the drawing units.
        
        Args:
            units: The units to use ('METERS', 'FEET', etc.)
        """
        self.units = units
        # In a real implementation, we would update the DXF file's units setting
    
    def get_all_elements(self) -> Dict[str, List[Any]]:
        """
        Get all elements from the drawing by layer.
        
        Returns:
            Dictionary with layer names as keys and lists of entities as values
        """
        if self.drawing is None:
            raise ValueError("No drawing available")
        
        msp = self.drawing.modelspace()
        elements = {}
        
        # Group elements by layer
        for entity in msp:
            layer = entity.dxf.layer
            if layer not in elements:
                elements[layer] = []
            elements[layer].append(entity)
        
        return elements
    
    def clear_layer(self, layer_name: str) -> None:
        """
        Clear all entities from a specified layer.
        
        Args:
            layer_name: Name of the layer to clear
        """
        if self.drawing is None:
            raise ValueError("No drawing available")
        
        if layer_name not in self.drawing.layers:
            return
        
        msp = self.drawing.modelspace()
        
        # Find all entities in this layer
        to_delete = [e for e in msp if e.dxf.layer == layer_name]
        
        # Delete them
        for entity in to_delete:
            msp.delete_entity(entity)