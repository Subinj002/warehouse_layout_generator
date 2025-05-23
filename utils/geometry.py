"""
geometry.py - Geometric utility functions for warehouse layout generation

This module contains utility functions for geometric calculations and operations
needed for warehouse layout design, including:
- Point, line, and polygon operations
- Distance and intersection calculations
- Area optimization algorithms
- Collision detection
- Layout alignment utilities
"""

import math
import numpy as np
from typing import List, Tuple, Dict, Union, Optional

# Type aliases for clarity
Point = Tuple[float, float]
Line = Tuple[Point, Point]
Rect = Tuple[Point, Point]  # (top-left, bottom-right)
Polygon = List[Point]

"""
Geometry utilities for warehouse layout calculations.
"""
from dataclasses import dataclass
import math

@dataclass
class Rectangle:
    """Represents a rectangle in 2D space."""
    x: float
    y: float
    width: float
    height: float

def calculate_distance(point1: tuple[float, float], point2: tuple[float, float]) -> float:
    """Calculate Euclidean distance between two points."""
    return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

def check_overlap(rect1: Rectangle, rect2: Rectangle) -> bool:
    """Check if two rectangles overlap."""
    return not (rect1.x + rect1.width < rect2.x or
                rect2.x + rect2.width < rect1.x or
                rect1.y + rect1.height < rect2.y or
                rect2.y + rect2.height < rect1.y)

def distance(p1: Point, p2: Point) -> float:
    """Calculate Euclidean distance between two points."""
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


def manhattan_distance(p1: Point, p2: Point) -> float:
    """Calculate Manhattan (L1) distance between two points."""
    return abs(p2[0] - p1[0]) + abs(p2[1] - p1[1])


def midpoint(p1: Point, p2: Point) -> Point:
    """Calculate the midpoint between two points."""
    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)


def angle_between_points(p1: Point, p2: Point) -> float:
    """Calculate the angle (in radians) between two points relative to horizontal."""
    return math.atan2(p2[1] - p1[1], p2[0] - p1[0])


def rotate_point(point: Point, origin: Point, angle: float) -> Point:
    """
    Rotate a point around an origin by a specified angle (in radians).
    
    Args:
        point: The point to rotate
        origin: The origin to rotate around
        angle: The angle in radians
        
    Returns:
        The rotated point
    """
    ox, oy = origin
    px, py = point
    
    # Translate point to origin
    qx = px - ox
    qy = py - oy
    
    # Rotate
    rx = qx * math.cos(angle) - qy * math.sin(angle)
    ry = qx * math.sin(angle) + qy * math.cos(angle)
    
    # Translate back
    return (rx + ox, ry + oy)


def rotate_polygon(polygon: Polygon, origin: Point, angle: float) -> Polygon:
    """Rotate a polygon around an origin by a specified angle (in radians)."""
    return [rotate_point(p, origin, angle) for p in polygon]


def point_in_rect(point: Point, rect: Rect) -> bool:
    """Check if a point is inside a rectangle."""
    (x, y) = point
    (x1, y1), (x2, y2) = rect
    
    # Ensure x1 <= x2 and y1 <= y2
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
        
    return x1 <= x <= x2 and y1 <= y <= y2


def rect_area(rect: Rect) -> float:
    """Calculate the area of a rectangle."""
    (x1, y1), (x2, y2) = rect
    return abs((x2 - x1) * (y2 - y1))


def rect_perimeter(rect: Rect) -> float:
    """Calculate the perimeter of a rectangle."""
    (x1, y1), (x2, y2) = rect
    width = abs(x2 - x1)
    height = abs(y2 - y1)
    return 2 * (width + height)


def rects_overlap(rect1: Rect, rect2: Rect) -> bool:
    """Check if two rectangles overlap."""
    (x1, y1), (x2, y2) = rect1
    (x3, y3), (x4, y4) = rect2
    
    # Ensure coordinates are in the right order
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    if x3 > x4:
        x3, x4 = x4, x3
    if y3 > y4:
        y3, y4 = y4, y3
    
    # Check for overlap
    return not (x2 < x3 or x4 < x1 or y2 < y3 or y4 < y1)


def rect_to_polygon(rect: Rect) -> Polygon:
    """Convert a rectangle to a polygon (list of points)."""
    (x1, y1), (x2, y2) = rect
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


def polygon_area(polygon: Polygon) -> float:
    """
    Calculate the area of a polygon using the Shoelace formula.
    
    Args:
        polygon: List of points representing the polygon vertices
        
    Returns:
        Area of the polygon
    """
    if len(polygon) < 3:
        return 0.0
        
    area = 0.0
    for i in range(len(polygon)):
        j = (i + 1) % len(polygon)
        area += polygon[i][0] * polygon[j][1]
        area -= polygon[j][0] * polygon[i][1]
    
    return abs(area) / 2.0


def polygon_centroid(polygon: Polygon) -> Point:
    """Calculate the centroid (geometric center) of a polygon."""
    if len(polygon) < 3:
        return polygon[0] if polygon else (0, 0)
    
    area = polygon_area(polygon)
    if area == 0:
        return polygon[0]
    
    cx = cy = 0
    for i in range(len(polygon)):
        j = (i + 1) % len(polygon)
        factor = polygon[i][0] * polygon[j][1] - polygon[j][0] * polygon[i][1]
        cx += (polygon[i][0] + polygon[j][0]) * factor
        cy += (polygon[i][1] + polygon[j][1]) * factor
    
    return (cx / (6 * area), cy / (6 * area))


def point_in_polygon(point: Point, polygon: Polygon) -> bool:
    """
    Check if a point is inside a polygon using the ray casting algorithm.
    
    Args:
        point: The point to check
        polygon: List of points representing the polygon vertices
        
    Returns:
        True if the point is inside the polygon, False otherwise
    """
    if len(polygon) < 3:
        return False
        
    x, y = point
    n = len(polygon)
    inside = False
    
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside


def line_intersection(line1: Line, line2: Line) -> Optional[Point]:
    """
    Calculate the intersection point of two line segments if it exists.
    
    Args:
        line1: First line segment defined by two points
        line2: Second line segment defined by two points
        
    Returns:
        The intersection point if lines intersect, None otherwise
    """
    ((x1, y1), (x2, y2)) = line1
    ((x3, y3), (x4, y4)) = line2
    
    # Calculate determinants
    denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
    
    # If lines are parallel
    if denom == 0:
        return None
        
    ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
    ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denom
    
    # Check if intersection is within both line segments
    if 0 <= ua <= 1 and 0 <= ub <= 1:
        x = x1 + ua * (x2 - x1)
        y = y1 + ua * (y2 - y1)
        return (x, y)
    
    return None


def grid_snap(point: Point, grid_size: float) -> Point:
    """
    Snap a point to the nearest grid intersection.
    
    Args:
        point: The point to snap
        grid_size: The grid size
        
    Returns:
        The snapped point
    """
    x, y = point
    return (round(x / grid_size) * grid_size, round(y / grid_size) * grid_size)


def align_rectangle_to_grid(rect: Rect, grid_size: float) -> Rect:
    """Align a rectangle to the nearest grid."""
    (x1, y1), (x2, y2) = rect
    return (grid_snap((x1, y1), grid_size), grid_snap((x2, y2), grid_size))


def bounding_box(points: List[Point]) -> Rect:
    """
    Calculate the axis-aligned bounding box for a set of points.
    
    Args:
        points: List of points
        
    Returns:
        The bounding box as a rectangle (top-left, bottom-right)
    """
    if not points:
        return ((0, 0), (0, 0))
        
    min_x = min(p[0] for p in points)
    min_y = min(p[1] for p in points)
    max_x = max(p[0] for p in points)
    max_y = max(p[1] for p in points)
    
    return ((min_x, min_y), (max_x, max_y))


def expand_rect(rect: Rect, amount: float) -> Rect:
    """
    Expand a rectangle in all directions by a specified amount.
    
    Args:
        rect: The rectangle to expand
        amount: The amount to expand by
        
    Returns:
        The expanded rectangle
    """
    (x1, y1), (x2, y2) = rect
    return ((x1 - amount, y1 - amount), (x2 + amount, y2 + amount))


def rect_union(rect1: Rect, rect2: Rect) -> Rect:
    """Calculate the union of two rectangles (smallest rectangle that contains both)."""
    (x1, y1), (x2, y2) = rect1
    (x3, y3), (x4, y4) = rect2
    
    min_x = min(x1, x3)
    min_y = min(y1, y3)
    max_x = max(x2, x4)
    max_y = max(y2, y4)
    
    return ((min_x, min_y), (max_x, max_y))


def rect_intersection(rect1: Rect, rect2: Rect) -> Optional[Rect]:
    """
    Calculate the intersection of two rectangles if they overlap.
    
    Args:
        rect1: First rectangle
        rect2: Second rectangle
        
    Returns:
        The intersection rectangle if rectangles overlap, None otherwise
    """
    (x1, y1), (x2, y2) = rect1
    (x3, y3), (x4, y4) = rect2
    
    # Ensure coordinates are in the right order
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    if x3 > x4:
        x3, x4 = x4, x3
    if y3 > y4:
        y3, y4 = y4, y3
    
    # Find intersection
    max_x1 = max(x1, x3)
    max_y1 = max(y1, y3)
    min_x2 = min(x2, x4)
    min_y2 = min(y2, y4)
    
    # Check if rectangles overlap
    if max_x1 < min_x2 and max_y1 < min_y2:
        return ((max_x1, max_y1), (min_x2, min_y2))
    
    return None


def generate_grid_points(rect: Rect, spacing: float) -> List[Point]:
    """
    Generate a grid of points within a rectangle with specified spacing.
    
    Args:
        rect: The rectangle defining the grid bounds
        spacing: The spacing between grid points
        
    Returns:
        List of grid points
    """
    (x1, y1), (x2, y2) = rect
    
    # Ensure coordinates are in the right order
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    
    grid_points = []
    x = x1
    while x <= x2:
        y = y1
        while y <= y2:
            grid_points.append((x, y))
            y += spacing
        x += spacing
    
    return grid_points


def convex_hull(points: List[Point]) -> Polygon:
    """
    Calculate the convex hull of a set of points using Graham scan algorithm.
    
    Args:
        points: List of points
        
    Returns:
        List of points forming the convex hull
    """
    if len(points) <= 3:
        return points
    
    # Find the point with the lowest y-coordinate
    start_point = min(points, key=lambda p: (p[1], p[0]))
    
    # Sort points by polar angle with respect to start_point
    def polar_angle(p):
        return math.atan2(p[1] - start_point[1], p[0] - start_point[0])
    
    sorted_points = sorted(points, key=polar_angle)
    
    # Graham scan algorithm
    hull = [start_point, sorted_points[0]]
    for point in sorted_points[1:]:
        while len(hull) > 1 and _cross_product(hull[-2], hull[-1], point) <= 0:
            hull.pop()
        hull.append(point)
    
    return hull


def _cross_product(p1: Point, p2: Point, p3: Point) -> float:
    """Helper function to compute the cross product (p2-p1) x (p3-p1)."""
    return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])


def get_rect_corners(rect: Rect) -> List[Point]:
    """Get the four corners of a rectangle."""
    (x1, y1), (x2, y2) = rect
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


def get_rect_dimensions(rect: Rect) -> Tuple[float, float]:
    """Get the width and height of a rectangle."""
    (x1, y1), (x2, y2) = rect
    return (abs(x2 - x1), abs(y2 - y1))


def rect_center(rect: Rect) -> Point:
    """Get the center point of a rectangle."""
    (x1, y1), (x2, y2) = rect
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def create_rect_from_center(center: Point, width: float, height: float) -> Rect:
    """Create a rectangle from center point and dimensions."""
    x, y = center
    half_width = width / 2
    half_height = height / 2
    return ((x - half_width, y - half_height), (x + half_width, y + half_height))


def rect_from_points(points: List[Point]) -> Rect:
    """Create the minimum bounding rectangle from a list of points."""
    min_x = min(p[0] for p in points)
    min_y = min(p[1] for p in points)
    max_x = max(p[0] for p in points)
    max_y = max(p[1] for p in points)
    return ((min_x, min_y), (max_x, max_y))


def minimum_distance_between_rects(rect1: Rect, rect2: Rect) -> float:
    """
    Calculate the minimum distance between two rectangles.
    If rectangles overlap, distance is 0.
    """
    (x1, y1), (x2, y2) = rect1
    (x3, y3), (x4, y4) = rect2
    
    # Ensure coordinates are in the right order
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    if x3 > x4:
        x3, x4 = x4, x3
    if y3 > y4:
        y3, y4 = y4, y3
    
    dx = max(0, max(x1 - x4, x3 - x2))
    dy = max(0, max(y1 - y4, y3 - y2))
    
    return math.sqrt(dx * dx + dy * dy)


def rectangles_touching(rect1: Rect, rect2: Rect, tolerance: float = 1e-6) -> bool:
    """Check if two rectangles are touching (sharing an edge or corner)."""
    return minimum_distance_between_rects(rect1, rect2) <= tolerance


def closest_point_on_rect(point: Point, rect: Rect) -> Point:
    """Find the closest point on a rectangle to a given point."""
    (x1, y1), (x2, y2) = rect
    x, y = point
    
    # Ensure coordinates are in the right order
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    
    # Find the closest x-coordinate
    if x < x1:
        closest_x = x1
    elif x > x2:
        closest_x = x2
    else:
        closest_x = x
    
    # Find the closest y-coordinate
    if y < y1:
        closest_y = y1
    elif y > y2:
        closest_y = y2
    else:
        closest_y = y
    
    return (closest_x, closest_y)


def path_length(path: List[Point]) -> float:
    """Calculate the total length of a path."""
    if len(path) < 2:
        return 0.0
        
    total_length = 0.0
    for i in range(len(path) - 1):
        total_length += distance(path[i], path[i + 1])
    
    return total_length
