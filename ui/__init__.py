"""
UI Module for Warehouse Layout Generator.

This module contains user interface components for the Warehouse Layout Generator,
including both GUI and CLI interfaces for configuring warehouse layouts,
visualizing the generated layouts, and exporting CAD files.

The UI layer serves as the bridge between the user and the core layout engine,
handling user inputs, parameter validation, and result presentation.
"""

from .cli import CLI
from .gui import GUI

__all__ = ['CLI', 'GUI']


def get_interface(interface_type='cli'):
    """
    Factory function to get the appropriate user interface.
    
    Args:
        interface_type (str): Type of interface to use ('cli' or 'gui')
        
    Returns:
        Interface object (CLI or GUI instance)
        
    Raises:
        ValueError: If an invalid interface type is provided
    """
    if interface_type.lower() == 'cli':
        return CLI()
    elif interface_type.lower() == 'gui':
        return GUI()
    else:
        raise ValueError(f"Invalid interface type: {interface_type}. Must be 'cli' or 'gui'")