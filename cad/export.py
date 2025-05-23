"""
Warehouse Layout Generator - Export Module

This module handles exporting warehouse layouts to various file formats including:
- DXF (for CAD software)
- SVG (for web/vector graphics)
- PDF (for documentation)
- JSON (for data interchange)

It provides a unified interface for all export operations and handles format-specific
considerations and optimizations.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

# Import local modules
from core.warehouse_elements import WarehouseLayout
from cad.ezdxf_interface import create_dxf_document, add_layout_to_dxf
from cad.drawing_utils import convert_to_svg

# Setup logging
logger = logging.getLogger(__name__)


class LayoutExporter:
    """Class for exporting warehouse layouts to various file formats."""
    
    SUPPORTED_FORMATS = ['dxf', 'svg', 'pdf', 'json']
    
    def __init__(self, layout: WarehouseLayout):
        """
        Initialize the exporter with a warehouse layout.
        
        Args:
            layout: The warehouse layout to export
        """
        self.layout = layout
        self.export_dir = None
    
    def set_export_directory(self, directory: Union[str, Path]) -> None:
        """
        Set the directory where exported files will be saved.
        
        Args:
            directory: Path to the export directory
        """
        if isinstance(directory, str):
            directory = Path(directory)
        
        # Create directory if it doesn't exist
        if not directory.exists():
            directory.mkdir(parents=True)
            logger.info(f"Created export directory: {directory}")
        
        self.export_dir = directory
    
    def export(self, filename: str, format_type: str, options: Optional[Dict[str, Any]] = None) -> str:
        """
        Export the layout to the specified format.
        
        Args:
            filename: Name of the output file (without extension)
            format_type: Format type ('dxf', 'svg', 'pdf', 'json')
            options: Format-specific export options
            
        Returns:
            Path to the exported file
            
        Raises:
            ValueError: If the format is not supported
        """
        if format_type.lower() not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {format_type}. "
                             f"Supported formats are: {', '.join(self.SUPPORTED_FORMATS)}")
        
        if self.export_dir is None:
            self.set_export_directory(Path.cwd() / "exports")
        
        options = options or {}
        
        # Call the appropriate export method based on format
        export_method = getattr(self, f"_export_to_{format_type.lower()}")
        output_path = export_method(filename, options)
        
        logger.info(f"Successfully exported layout to {output_path}")
        return output_path
    
    def _export_to_dxf(self, filename: str, options: Dict[str, Any]) -> str:
        """
        Export the layout to DXF format.
        
        Args:
            filename: Name of the output file
            options: DXF-specific export options
            
        Returns:
            Path to the exported DXF file
        """
        # Ensure filename has .dxf extension
        if not filename.lower().endswith('.dxf'):
            filename += '.dxf'
        
        output_path = self.export_dir / filename
        
        # Get DXF-specific options
        modelspace = options.get('modelspace', True)
        paperspace = options.get('paperspace', False)
        use_layouts = options.get('use_layouts', True)
        dxf_version = options.get('version', 'R2018')
        
        # Create the DXF document
        doc = create_dxf_document(dxf_version)
        
        # Add the warehouse layout to the DXF document
        add_layout_to_dxf(doc, self.layout, modelspace=modelspace, 
                          paperspace=paperspace, use_layouts=use_layouts)
        
        # Save the document
        doc.saveas(str(output_path))
        
        return str(output_path)
    
    def _export_to_svg(self, filename: str, options: Dict[str, Any]) -> str:
        """
        Export the layout to SVG format.
        
        Args:
            filename: Name of the output file
            options: SVG-specific export options
            
        Returns:
            Path to the exported SVG file
        """
        # Ensure filename has .svg extension
        if not filename.lower().endswith('.svg'):
            filename += '.svg'
        
        output_path = self.export_dir / filename
        
        # Get SVG-specific options
        scale = options.get('scale', 1.0)
        include_grid = options.get('include_grid', True)
        include_dimensions = options.get('include_dimensions', True)
        
        # Convert the layout to SVG
        svg_content = convert_to_svg(
            self.layout, 
            scale=scale,
            include_grid=include_grid, 
            include_dimensions=include_dimensions
        )
        
        # Write the SVG content to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(svg_content)
        
        return str(output_path)
    
    def _export_to_pdf(self, filename: str, options: Dict[str, Any]) -> str:
        """
        Export the layout to PDF format.
        
        Args:
            filename: Name of the output file
            options: PDF-specific export options
            
        Returns:
            Path to the exported PDF file
        """
        # Ensure filename has .pdf extension
        if not filename.lower().endswith('.pdf'):
            filename += '.pdf'
        
        output_path = self.export_dir / filename
        
        # Get PDF-specific options
        paper_size = options.get('paper_size', 'A4')
        orientation = options.get('orientation', 'landscape')
        scale = options.get('scale', 'fit')
        
        # For now, we'll create a PDF by first exporting to SVG and then converting
        # This is a placeholder implementation
        svg_filename = f"{filename.rsplit('.', 1)[0]}_temp.svg"
        svg_path = self._export_to_svg(svg_filename, {
            'scale': 1.0,
            'include_grid': True,
            'include_dimensions': True
        })
        
        # TODO: Implement proper SVG to PDF conversion
        # Here you would add code to convert the SVG to PDF with the specified options
        # This might involve using a library like Cairo, reportlab, or calling an external tool
        
        logger.warning("PDF export is not fully implemented yet. Using DXF export as fallback.")
        return self._export_to_dxf(filename.replace('.pdf', '.dxf'), options)
    
    def _export_to_json(self, filename: str, options: Dict[str, Any]) -> str:
        """
        Export the layout to JSON format.
        
        Args:
            filename: Name of the output file
            options: JSON-specific export options
            
        Returns:
            Path to the exported JSON file
        """
        # Ensure filename has .json extension
        if not filename.lower().endswith('.json'):
            filename += '.json'
        
        output_path = self.export_dir / filename
        
        # Get JSON-specific options
        pretty_print = options.get('pretty_print', True)
        include_metadata = options.get('include_metadata', True)
        
        # Convert the layout to a dictionary
        layout_data = self.layout.to_dict()
        
        # Add metadata if requested
        if include_metadata:
            layout_data['metadata'] = {
                'generated_at': self.layout.timestamp.isoformat() if hasattr(self.layout, 'timestamp') else None,
                'version': getattr(self.layout, 'version', '1.0'),
                'name': getattr(self.layout, 'name', 'Unnamed Layout'),
                'description': getattr(self.layout, 'description', '')
            }
        
        # Write the JSON data to file
        with open(output_path, 'w', encoding='utf-8') as f:
            if pretty_print:
                json.dump(layout_data, f, indent=2)
            else:
                json.dump(layout_data, f)
        
        return str(output_path)


def batch_export(layout: WarehouseLayout, 
                formats: List[str], 
                base_filename: str, 
                export_dir: Optional[Union[str, Path]] = None,
                options: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, str]:
    """
    Export a layout to multiple formats at once.
    
    Args:
        layout: The warehouse layout to export
        formats: List of formats to export to
        base_filename: Base name for the output files
        export_dir: Directory to save the exported files
        options: Dictionary mapping format names to format-specific options
        
    Returns:
        Dictionary mapping format names to exported file paths
    """
    exporter = LayoutExporter(layout)
    
    if export_dir:
        exporter.set_export_directory(export_dir)
    
    options = options or {}
    results = {}
    
    for fmt in formats:
        if fmt.lower() not in LayoutExporter.SUPPORTED_FORMATS:
            logger.warning(f"Skipping unsupported format: {fmt}")
            continue
        
        fmt_options = options.get(fmt, {})
        filename = f"{base_filename}.{fmt.lower()}"
        
        try:
            output_path = exporter.export(filename, fmt, fmt_options)
            results[fmt] = output_path
        except Exception as e:
            logger.error(f"Failed to export to {fmt}: {e}")
            results[fmt] = None
    
    return results