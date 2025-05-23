"""
drawing_utils.py - Utility functions for drawing warehouse elements in CAD formats

This module provides helper functions for creating common warehouse elements
like racks, aisles, zones, and other components in CAD drawings.
It abstracts the complexity of drawing these elements and provides a clean
interface for the layout engine to use.
"""

import math
from typing import Dict, List, Tuple, Union, Optional


def create_rectangle(
    x: float, y: float, width: float, height: float, 
    rotation: float = 0, layer: str = "0", 
    color: int = None, linetype: str = None
) -> Dict:
    """
    Create a rectangle with the given parameters.
    
    Args:
        x: X-coordinate of the bottom-left corner
        y: Y-coordinate of the bottom-left corner
        width: Width of the rectangle
        height: Height of the rectangle
        rotation: Rotation angle in degrees (counterclockwise)
        layer: Layer name
        color: Color number (AutoCAD color index)
        linetype: Line type name
    
    Returns:
        Dictionary with rectangle properties and coordinates
    """
    # Convert rotation to radians
    rotation_rad = math.radians(rotation)
    
    # Calculate the four corners
    corners = [
        (x, y),  # Bottom-left
        (x + width, y),  # Bottom-right
        (x + width, y + height),  # Top-right
        (x, y + height),  # Top-left
    ]
    
    # Apply rotation if needed
    if rotation != 0:
        # Find center point
        center_x = x + width / 2
        center_y = y + height / 2
        
        # Rotate each corner around the center
        rotated_corners = []
        for cx, cy in corners:
            # Translate to origin
            tx = cx - center_x
            ty = cy - center_y
            
            # Rotate
            rx = tx * math.cos(rotation_rad) - ty * math.sin(rotation_rad)
            ry = tx * math.sin(rotation_rad) + ty * math.cos(rotation_rad)
            
            # Translate back
            rotated_corners.append((rx + center_x, ry + center_y))
        
        corners = rotated_corners
    
    return {
        "type": "polyline",
        "points": corners + [corners[0]],  # Close the loop
        "layer": layer,
        "color": color,
        "linetype": linetype,
        "width": width,
        "height": height,
        "rotation": rotation,
        "center": (x + width / 2, y + height / 2)
    }


def create_line(
    start_x: float, start_y: float, end_x: float, end_y: float,
    layer: str = "0", color: int = None, linetype: str = None, lineweight: float = None
) -> Dict:
    """
    Create a line from start to end points.
    
    Args:
        start_x: X-coordinate of start point
        start_y: Y-coordinate of start point
        end_x: X-coordinate of end point
        end_y: Y-coordinate of end point
        layer: Layer name
        color: Color number (AutoCAD color index)
        linetype: Line type name
        lineweight: Line weight/thickness
    
    Returns:
        Dictionary with line properties
    """
    return {
        "type": "line",
        "start": (start_x, start_y),
        "end": (end_x, end_y),
        "layer": layer,
        "color": color,
        "linetype": linetype,
        "lineweight": lineweight
    }


def create_circle(
    center_x: float, center_y: float, radius: float,
    layer: str = "0", color: int = None, linetype: str = None
) -> Dict:
    """
    Create a circle with given center and radius.
    
    Args:
        center_x: X-coordinate of center
        center_y: Y-coordinate of center
        radius: Circle radius
        layer: Layer name
        color: Color number (AutoCAD color index)
        linetype: Line type name
    
    Returns:
        Dictionary with circle properties
    """
    return {
        "type": "circle",
        "center": (center_x, center_y),
        "radius": radius,
        "layer": layer,
        "color": color,
        "linetype": linetype
    }


def create_arc(
    center_x: float, center_y: float, radius: float,
    start_angle: float, end_angle: float,
    layer: str = "0", color: int = None, linetype: str = None
) -> Dict:
    """
    Create an arc with given center, radius and angles.
    
    Args:
        center_x: X-coordinate of center
        center_y: Y-coordinate of center
        radius: Arc radius
        start_angle: Start angle in degrees
        end_angle: End angle in degrees
        layer: Layer name
        color: Color number (AutoCAD color index)
        linetype: Line type name
    
    Returns:
        Dictionary with arc properties
    """
    return {
        "type": "arc",
        "center": (center_x, center_y),
        "radius": radius,
        "start_angle": start_angle,
        "end_angle": end_angle,
        "layer": layer,
        "color": color,
        "linetype": linetype
    }


def create_text(
    x: float, y: float, text: str, height: float,
    rotation: float = 0, alignment: str = "left",
    layer: str = "0", color: int = None, style: str = "Standard"
) -> Dict:
    """
    Create a text entity.
    
    Args:
        x: X-coordinate of text insertion point
        y: Y-coordinate of text insertion point
        text: Text content
        height: Text height
        rotation: Rotation angle in degrees
        alignment: Text alignment ("left", "center", "right")
        layer: Layer name
        color: Color number (AutoCAD color index)
        style: Text style name
    
    Returns:
        Dictionary with text properties
    """
    return {
        "type": "text",
        "position": (x, y),
        "content": text,
        "height": height,
        "rotation": rotation,
        "alignment": alignment,
        "layer": layer,
        "color": color,
        "style": style
    }


def create_pallet_rack(
    x: float, y: float, width: float, depth: float, 
    levels: int = 3, bay_width: float = 2.7,
    rotation: float = 0, layer: str = "RACKS"
) -> List[Dict]:
    """
    Create a pallet rack with multiple bays.
    
    Args:
        x: X-coordinate of bottom-left corner
        y: Y-coordinate of bottom-left corner
        width: Total width of the rack
        depth: Depth of the rack
        levels: Number of vertical levels
        bay_width: Width of each bay
        rotation: Rotation angle in degrees
        layer: Layer name
    
    Returns:
        List of dictionaries representing the rack elements
    """
    elements = []
    
    # Calculate number of bays and uprights
    num_bays = max(1, int(width / bay_width))
    actual_bay_width = width / num_bays
    
    # Create the main rack outline
    rack_outline = create_rectangle(
        x, y, width, depth, rotation, layer, color=5
    )
    elements.append(rack_outline)
    
    # Create uprights
    for i in range(num_bays + 1):
        upright_x = x + i * actual_bay_width
        upright = create_rectangle(
            upright_x - 0.05, y, 0.1, depth, rotation, layer, color=1
        )
        elements.append(upright)
    
    # Create level lines
    level_height = depth / (levels + 1)
    for i in range(1, levels + 1):
        level_y = y + i * level_height
        level_line = create_line(
            x, level_y, x + width, level_y, layer, color=3, linetype="DASHED"
        )
        elements.append(level_line)
        
    # Add text label
    label = create_text(
        x + width / 2, y - 0.3, f"Rack {levels}L x {num_bays}B", 
        height=0.15, alignment="center", layer=layer, color=7
    )
    elements.append(label)
    
    return elements


def create_aisle(
    x: float, y: float, length: float, width: float,
    rotation: float = 0, layer: str = "AISLES",
    label: Optional[str] = None, hatch_pattern: Optional[str] = None
) -> List[Dict]:
    """
    Create an aisle with optional labels and hatching.
    
    Args:
        x: X-coordinate of start point
        y: Y-coordinate of start point
        length: Length of the aisle
        width: Width of the aisle
        rotation: Rotation angle in degrees
        layer: Layer name
        label: Optional label for the aisle
        hatch_pattern: Optional hatch pattern name
    
    Returns:
        List of dictionaries representing the aisle elements
    """
    elements = []
    
    # Create the aisle rectangle
    aisle = create_rectangle(
        x, y, length, width, rotation, layer, color=3, linetype="CONTINUOUS"
    )
    elements.append(aisle)
    
    # Add hatching if specified
    if hatch_pattern:
        elements.append({
            "type": "hatch",
            "boundary": aisle["points"][:-1],  # Don't duplicate the last point
            "pattern": hatch_pattern,
            "layer": layer,
            "color": 8
        })
    
    # Add label if specified
    if label:
        text = create_text(
            x + length / 2, y + width / 2, label,
            height=width * 0.3, alignment="center", layer=layer, color=7
        )
        elements.append(text)
    
    # Add directional arrows
    arrow_length = min(length, width) * 0.4
    center_x = x + length / 2
    center_y = y + width / 2
    
    arrow1 = create_line(
        center_x - arrow_length / 2, center_y,
        center_x + arrow_length / 2, center_y,
        layer, color=2, linetype="CONTINUOUS"
    )
    elements.append(arrow1)
    
    # Arrow head
    angle = math.radians(rotation)
    head_x = center_x + arrow_length / 2
    head_y = center_y
    
    arrow_head = [
        (head_x, head_y),
        (head_x - arrow_length * 0.2, head_y + arrow_length * 0.1),
        (head_x - arrow_length * 0.2, head_y - arrow_length * 0.1),
        (head_x, head_y)
    ]
    
    elements.append({
        "type": "polyline",
        "points": arrow_head,
        "layer": layer,
        "color": 2,
        "filled": True
    })
    
    return elements


def create_dock_door(
    x: float, y: float, width: float, depth: float,
    rotation: float = 0, door_number: Optional[str] = None,
    layer: str = "DOCKS"
) -> List[Dict]:
    """
    Create a dock door with optional numbering.
    
    Args:
        x: X-coordinate of the door position
        y: Y-coordinate of the door position
        width: Width of the door
        depth: Depth of the door area
        rotation: Rotation angle in degrees
        door_number: Optional door identifier
        layer: Layer name
    
    Returns:
        List of dictionaries representing the dock door elements
    """
    elements = []
    
    # Create the main door rectangle
    door = create_rectangle(
        x, y, width, depth, rotation, layer, color=1
    )
    elements.append(door)
    
    # Create the door opening line (thicker)
    door_line = create_line(
        x, y, x + width, y, layer, color=1, lineweight=0.5
    )
    elements.append(door_line)
    
    # Create dock leveler
    leveler_width = width * 0.8
    leveler_depth = depth * 0.5
    leveler_x = x + (width - leveler_width) / 2
    leveler_y = y + depth * 0.3
    
    leveler = create_rectangle(
        leveler_x, leveler_y, leveler_width, leveler_depth, 
        rotation, layer, color=5, linetype="DASHED"
    )
    elements.append(leveler)
    
    # Add door number if specified
    if door_number:
        text = create_text(
            x + width / 2, y - depth * 0.2, f"DOOR {door_number}",
            height=width * 0.15, alignment="center", layer=layer, color=7
        )
        elements.append(text)
    
    return elements


def create_zone(
    x: float, y: float, width: float, height: float,
    zone_type: str, rotation: float = 0,
    hatch_pattern: Optional[str] = None, layer: str = "ZONES"
) -> List[Dict]:
    """
    Create a warehouse zone (receiving, shipping, staging, etc.).
    
    Args:
        x: X-coordinate of bottom-left corner
        y: Y-coordinate of bottom-left corner
        width: Width of the zone
        height: Height of the zone
        zone_type: Type identifier (e.g., "RECEIVING", "SHIPPING")
        rotation: Rotation angle in degrees
        hatch_pattern: Optional hatch pattern name
        layer: Layer name
    
    Returns:
        List of dictionaries representing the zone elements
    """
    elements = []
    
    # Create zone outline with dashed lines
    zone_outline = create_rectangle(
        x, y, width, height, rotation, layer, color=4, linetype="DASHDOT"
    )
    elements.append(zone_outline)
    
    # Add hatching if specified
    if hatch_pattern:
        elements.append({
            "type": "hatch",
            "boundary": zone_outline["points"][:-1],
            "pattern": hatch_pattern,
            "layer": layer,
            "color": 4,
            "transparency": 50  # Semi-transparent
        })
    
    # Add zone label
    label = create_text(
        x + width / 2, y + height / 2, zone_type,
        height=min(width, height) * 0.1, alignment="center", 
        layer=layer, color=7
    )
    elements.append(label)
    
    return elements


def create_column(
    x: float, y: float, size: float, 
    layer: str = "COLUMNS", is_circular: bool = False
) -> Dict:
    """
    Create a building column.
    
    Args:
        x: X-coordinate of column center
        y: Y-coordinate of column center
        size: Size of the column (diameter for circular, side length for square)
        layer: Layer name
        is_circular: Whether the column is circular (True) or square (False)
    
    Returns:
        Dictionary representing the column
    """
    if is_circular:
        return create_circle(x, y, size / 2, layer, color=6)
    else:
        half_size = size / 2
        return create_rectangle(
            x - half_size, y - half_size, size, size, 0, layer, color=6
        )


def create_column_grid(
    start_x: float, start_y: float, 
    cols: int, rows: int, 
    col_spacing: float, row_spacing: float,
    column_size: float = 0.4, layer: str = "COLUMNS",
    is_circular: bool = False
) -> List[Dict]:
    """
    Create a grid of building columns.
    
    Args:
        start_x: X-coordinate of the first column
        start_y: Y-coordinate of the first column
        cols: Number of columns in X direction
        rows: Number of columns in Y direction
        col_spacing: Spacing between columns in X direction
        row_spacing: Spacing between columns in Y direction
        column_size: Size of each column
        layer: Layer name
        is_circular: Whether the columns are circular (True) or square (False)
    
    Returns:
        List of dictionaries representing the columns
    """
    elements = []
    
    for i in range(cols):
        for j in range(rows):
            x = start_x + i * col_spacing
            y = start_y + j * row_spacing
            
            column = create_column(x, y, column_size, layer, is_circular)
            column["id"] = f"COL-{chr(65 + i)}{j+1}"  # Add column ID (e.g., COL-A1)
            elements.append(column)
    
    return elements


def create_dimensions(
    start_x: float, start_y: float, end_x: float, end_y: float,
    offset: float = 1.0, layer: str = "DIMENSIONS",
    text_height: float = 0.2, precision: int = 2
) -> List[Dict]:
    """
    Create a linear dimension between two points.
    
    Args:
        start_x: X-coordinate of start point
        start_y: Y-coordinate of start point
        end_x: X-coordinate of end point
        end_y: Y-coordinate of end point
        offset: Perpendicular distance from the measured line
        layer: Layer name
        text_height: Height of dimension text
        precision: Number of decimal places for dimension value
    
    Returns:
        List of dictionaries representing the dimension elements
    """
    elements = []
    
    # Calculate the vector between points
    dx = end_x - start_x
    dy = end_y - start_y
    length = math.sqrt(dx**2 + dy**2)
    
    # Calculate perpendicular direction
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        # Points are too close, avoid division by zero
        return elements
    
    # Normalize and get perpendicular
    nx = dx / length
    ny = dy / length
    px, py = -ny, nx  # Perpendicular vector
    
    # Calculate offset points
    s_off_x = start_x + px * offset
    s_off_y = start_y + py * offset
    e_off_x = end_x + px * offset
    e_off_y = end_y + py * offset
    
    # Create dimension line
    dim_line = create_line(s_off_x, s_off_y, e_off_x, e_off_y, layer, color=2)
    elements.append(dim_line)
    
    # Create extension lines
    ext_line1 = create_line(start_x, start_y, s_off_x, s_off_y, layer, color=2, linetype="DASHED")
    ext_line2 = create_line(end_x, end_y, e_off_x, e_off_y, layer, color=2, linetype="DASHED")
    elements.append(ext_line1)
    elements.append(ext_line2)
    
    # Add dimension text
    dim_text = f"{length:.{precision}f}"
    mid_x = (s_off_x + e_off_x) / 2
    mid_y = (s_off_y + e_off_y) / 2
    
    # Calculate text rotation
    if abs(dx) < 1e-6:  # Vertical dimension
        text_rotation = 90
    else:
        text_rotation = math.degrees(math.atan2(dy, dx))
        # Make sure text is readable (not upside down)
        if text_rotation > 90 or text_rotation < -90:
            text_rotation += 180
            if text_rotation > 180:
                text_rotation -= 360
    
    text = create_text(
        mid_x, mid_y, dim_text, 
        height=text_height, rotation=text_rotation, alignment="center",
        layer=layer, color=7
    )
    elements.append(text)
    
    return elements


def create_north_arrow(
    x: float, y: float, size: float = 1.0, 
    rotation: float = 0, layer: str = "SYMBOLS"
) -> List[Dict]:
    """
    Create a north arrow symbol.
    
    Args:
        x: X-coordinate of symbol center
        y: Y-coordinate of symbol center
        size: Size of the symbol
        rotation: Rotation angle in degrees (0 = North is up)
        layer: Layer name
    
    Returns:
        List of dictionaries representing the north arrow elements
    """
    elements = []
    
    # Create circle
    circle = create_circle(x, y, size * 0.4, layer, color=5)
    elements.append(circle)
    
    # Create arrow
    arrow_length = size * 0.7
    
    # Apply rotation (default is north = up)
    angle_rad = math.radians(rotation - 90)  # -90 because 0 degrees is east in mathematics
    arrow_end_x = x + arrow_length * math.cos(angle_rad)
    arrow_end_y = y + arrow_length * math.sin(angle_rad)
    
    # Main arrow line
    arrow_line = create_line(x, y, arrow_end_x, arrow_end_y, layer, color=1, lineweight=0.25)
    elements.append(arrow_line)
    
    # Arrow head
    head_size = size * 0.15
    angle1 = angle_rad + math.radians(150)
    angle2 = angle_rad - math.radians(150)
    
    head1_x = arrow_end_x + head_size * math.cos(angle1)
    head1_y = arrow_end_y + head_size * math.sin(angle1)
    head2_x = arrow_end_x + head_size * math.cos(angle2)
    head2_y = arrow_end_y + head_size * math.sin(angle2)
    
    head_line1 = create_line(arrow_end_x, arrow_end_y, head1_x, head1_y, layer, color=1)
    head_line2 = create_line(arrow_end_x, arrow_end_y, head2_x, head2_y, layer, color=1)
    elements.append(head_line1)
    elements.append(head_line2)
    
    # Add "N" text
    n_text = create_text(
        x, y - size * 0.6, "N", 
        height=size * 0.25, alignment="center", 
        layer=layer, color=7
    )
    elements.append(n_text)
    
    return elements


def create_scale_bar(
    x: float, y: float, width: float, 
    scale: float = 100, units: str = "m", 
    layer: str = "SYMBOLS"
) -> List[Dict]:
    """
    Create a scale bar.
    
    Args:
        x: X-coordinate of bottom-left corner
        y: Y-coordinate of bottom-left corner
        width: Width of the scale bar in drawing units
        scale: Scale factor (e.g., 100 means 1:100)
        units: Units label (e.g., "m" for meters)
        layer: Layer name
    
    Returns:
        List of dictionaries representing the scale bar elements
    """
    elements = []
    
    # Calculate segments (5 segments alternating black/white)
    segment_width = width / 5
    height = segment_width * 0.5
    
    # Create the segments
    for i in range(5):
        segment_x = x + i * segment_width
        color = 7 if i % 2 == 0 else 0  # Alternate between white and black
        
        segment = create_rectangle(
            segment_x, y, segment_width, height, 0, layer, color=color
        )
        elements.append(segment)
    
    # Add labels
    real_width = width * scale  # Real-world width
    interval = real_width / 5
    
    # Start label
    start_label = create_text(
        x, y - height * 0.5, "0", 
        height=height * 0.7, alignment="left", 
        layer=layer, color=7
    )
    elements.append(start_label)
    
    # Middle labels
    for i in range(1, 5):
        label_x = x + i * segment_width
        label_value = i * interval
        
        # Format the label (no decimal for whole numbers)
        if label_value.is_integer():
            label_text = f"{int(label_value)}"
        else:
            label_text = f"{label_value:.1f}"
        
        middle_label = create_text(
            label_x, y - height * 0.5, label_text, 
            height=height * 0.7, alignment="center", 
            layer=layer, color=7
        )
        elements.append(middle_label)
    
    # Scale label
    scale_label = create_text(
        x + width / 2, y - height * 1.5, f"SCALE 1:{int(scale)} ({units})", 
        height=height * 0.7, alignment="center", 
        layer=layer, color=7
    )
    elements.append(scale_label)
    
    return elements


def create_forklift(
    x: float, y: float, rotation: float = 0, 
    layer: str = "EQUIPMENT"
) -> List[Dict]:
    """
    Create a simplified forklift symbol.
    
    Args:
        x: X-coordinate of the forklift center
        y: Y-coordinate of the forklift center
        rotation: Rotation angle in degrees
        layer: Layer name
    
    Returns:
        List of dictionaries representing the forklift elements
    """
    elements = []
    
    # Forklift dimensions
    length = 2.0
    width = 1.2
    
    # Create main body
    body = create_rectangle(
        x - length / 2, y - width / 2, length, width, 
        rotation, layer, color=3
    )
    elements.append(body)
    
    # Add forks (simplified)
    # We need to calculate the front of the forklift based on rotation
    angle_rad = math.radians(rotation)
    front_x = x + (length / 2) * math.cos(angle_rad)
    front_y = y + (length / 2) * math.sin(angle_rad)
    
    # Fork length
    fork_length = 1.0
    fork_width = 0.15
    fork_spacing = 0.6
    
    # Left fork
    fork1_start_x = front_x
    fork1_start_y = front_y - fork_spacing / 2
    fork1_end_x = front_x + fork_length * math.cos(angle_rad)
    fork1_end_y = front_y - fork_spacing / 2 + fork_length * math.sin(angle_rad)
    
    # Create left fork as a rectangle
    fork1 = create_rectangle(
        fork1_start_x - fork_width / 2, fork1_start_y - fork_width / 2,
        fork_length, fork_width, rotation, layer, color=1
    )
    elements.append(fork1)
    
    # Right fork
    fork2_start_x = front_x
    fork2_start_y = front_y + fork_spacing / 2
    fork2_end_x = front_x + fork_length * math.cos(angle_rad)
    fork2_end_y = front_y + fork_spacing / 2 + fork_length * math.sin(angle_rad)
    
    # Create right fork as a rectangle
    fork2 = create_rectangle(
        fork2_start_x - fork_width / 2, fork2_start_y - fork_width / 2,
        fork_length, fork_width, rotation, layer, color=1
    )
    elements.append(fork2)
    
    # Add driver's position indicator (circle)
    back_x = x - (length / 2 - 0.5) * math.cos(angle_rad)
    back_y = y - (length / 2 - 0.5) * math.sin(angle_rad)
    
    driver = create_circle(back_x, back_y, 0.3, layer, color=5)
    elements.append(driver)
    
    # Add "FORK" label
    label = create_text(
        x, y, "FL", height=0.4, rotation=rotation,
        alignment="center", layer=layer, color=7
    )
    elements.append(label)
    
    return elements