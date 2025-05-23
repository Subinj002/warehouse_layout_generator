#!/usr/bin/env python3
"""
Command Line Interface for the Warehouse Layout Generator.
This module provides a CLI for interacting with the warehouse layout generator.
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.layout_engine import LayoutEngine
from core.warehouse_elements import WarehouseElements
from config.config_schema import validate_config
from cad.export import export_layout
from utils.helpers import setup_logging

# Setup logging
logger = logging.getLogger(__name__)


class WarehouseLayoutCLI:
    """Command Line Interface for the Warehouse Layout Generator."""

    def __init__(self):
        self.parser = self._create_parser()
        self.args = None
        self.config = None
        self.layout_engine = None

    def _create_parser(self):
        """Create the argument parser for the CLI."""
        parser = argparse.ArgumentParser(
            description="Warehouse Layout Generator CLI",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  Generate a layout using a custom config file:
    python -m ui.cli --config path/to/config.json

  Generate a layout and export to DXF:
    python -m ui.cli --config path/to/config.json --export output.dxf

  Run the optimization algorithm:
    python -m ui.cli --config path/to/config.json --optimize

  List available warehouse elements:
    python -m ui.cli --list-elements
"""
        )

        parser.add_argument(
            "--config", "-c",
            help="Path to the configuration file",
            type=str,
            default=str(Path(__file__).resolve().parent.parent / "config" / "default_config.json")
        )
        
        parser.add_argument(
            "--export", "-e",
            help="Export the layout to a DXF file",
            type=str
        )
        
        parser.add_argument(
            "--optimize", "-o",
            help="Run the optimization algorithm",
            action="store_true"
        )
        
        parser.add_argument(
            "--verbose", "-v",
            help="Increase output verbosity",
            action="count",
            default=0
        )
        
        parser.add_argument(
            "--list-elements",
            help="List available warehouse elements",
            action="store_true"
        )
        
        parser.add_argument(
            "--output", 
            help="Output directory for generated files",
            type=str,
            default="./output"
        )
        
        parser.add_argument(
            "--ai-model",
            help="Select AI model for layout generation",
            choices=["basic", "advanced", "expert"],
            default="basic"
        )
        
        return parser

    def _setup_logging(self):
        """Configure logging based on verbosity level."""
        log_levels = {
            0: logging.WARNING,
            1: logging.INFO,
            2: logging.DEBUG
        }
        verbosity = min(self.args.verbose, 2)  # Cap at 2 for DEBUG level
        setup_logging(log_levels[verbosity])
        logger.debug("Debug logging enabled")

    def _load_config(self):
        """Load and validate configuration file."""
        try:
            config_path = Path(self.args.config)
            if not config_path.exists():
                logger.error(f"Configuration file not found: {config_path}")
                sys.exit(1)
                
            with open(config_path, 'r') as f:
                self.config = json.load(f)
                
            # Validate the configuration
            validate_config(self.config)
            logger.info(f"Configuration loaded from {config_path}")
            
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in configuration file: {config_path}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            sys.exit(1)

    def _list_warehouse_elements(self):
        """Display a list of available warehouse elements."""
        elements = WarehouseElements.get_available_elements()
        print("\nAvailable Warehouse Elements:")
        print("=============================")
        
        for category, items in elements.items():
            print(f"\n{category.upper()}:")
            for item in items:
                print(f"  - {item['name']}: {item['description']}")
        
        print("\nUsage in config:")
        print("  Add elements to your configuration file using their names.")
        print("  Example: {'type': 'rack', 'name': 'standard_pallet_rack', ...}")
        
    def _ensure_output_directory(self):
        """Ensure the output directory exists."""
        output_dir = Path(self.args.output)
        if not output_dir.exists():
            output_dir.mkdir(parents=True)
            logger.info(f"Created output directory: {output_dir}")

    def _generate_layout(self):
        """Generate the warehouse layout using the configuration."""
        try:
            self.layout_engine = LayoutEngine(self.config)
            
            if self.args.optimize:
                logger.info("Running layout optimization...")
                from ai.space_optimizer import SpaceOptimizer
                optimizer = SpaceOptimizer(self.layout_engine, model_type=self.args.ai_model)
                layout = optimizer.optimize_layout()
                logger.info("Layout optimization completed")
            else:
                logger.info("Generating layout...")
                layout = self.layout_engine.generate_layout()
                logger.info("Layout generation completed")
            
            # If export is requested
            if self.args.export:
                export_path = Path(self.args.output) / self.args.export
                logger.info(f"Exporting layout to {export_path}")
                export_layout(layout, str(export_path), format="dxf")
                print(f"\nLayout exported successfully to {export_path}")
            
            return layout
            
        except Exception as e:
            logger.error(f"Error generating layout: {str(e)}")
            if self.args.verbose >= 2:  # If debug mode
                import traceback
                traceback.print_exc()
            sys.exit(1)

    def run(self):
        """Run the CLI application."""
        self.args = self.parser.parse_args()
        self._setup_logging()
        
        # Handle list elements command
        if self.args.list_elements:
            self._list_warehouse_elements()
            return
        
        # Load configuration
        self._load_config()
        
        # Ensure output directory exists
        self._ensure_output_directory()
        
        # Generate layout
        layout = self._generate_layout()
        
        print("\nWarehouse Layout Generation Complete!")
        print(f"- Total area: {layout.get_total_area()} sq ft")
        print(f"- Storage capacity: {layout.get_storage_capacity()} pallets")
        print(f"- Utilization: {layout.get_utilization_percentage():.2f}%")
        
        if not self.args.export:
            print("\nTip: Use --export option to save the layout to a DXF file.")


def main():
    """Entry point for the CLI application."""
    cli = WarehouseLayoutCLI()
    cli.run()


if __name__ == "__main__":
    main()