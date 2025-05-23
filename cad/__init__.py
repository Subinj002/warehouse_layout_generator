"""
CAD module for Warehouse Layout Generator.

This module handles CAD integration, including drawing generation, 
format conversion, and export capabilities for warehouse layouts.

Components:
    - ezdxf_interface: Interface for working with the ezdxf library to create DXF files
    - drawing_utils: Utility functions for drawing warehouse elements
    - export: Functions for exporting layouts to various CAD formats
"""

from .ezdxf_interface import setup_drawing, add_warehouse_boundaries, add_racks, add_aisles, add_staging_areas
from .drawing_utils import draw_rectangle, draw_line, draw_text, draw_dimension
from .export import export_to_dxf, export_to_pdf, export_to_png

__all__ = [
    # ezdxf_interface exports
    'setup_drawing',
    'add_warehouse_boundaries',
    'add_racks',
    'add_aisles',
    'add_staging_areas',
    
    # drawing_utils exports
    'draw_rectangle',
    'draw_line', 
    'draw_text',
    'draw_dimension',
    
    # export functions
    'export_to_dxf',
    'export_to_pdf',
    'export_to_png',
]

# Version of the CAD module
__version__ = '0.1.0'   