#!/usr/bin/env python
"""
Warehouse Layout Generator - Main Entry Point

This module serves as the main entry point for the Warehouse Layout Generator application.
It handles command-line arguments, initializes components, and orchestrates the overall workflow.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Import core modules
from core.layout_engine import LayoutEngine
from core.warehouse_elements import Warehouse
from config.config_schema import validate_config
from cad.export import export_to_dxf
from utils.helpers import setup_logging
from ai.layout_generator import LayoutGenerator
from ui.gui import launch_gui
from ui.cli import process_cli_commands


def load_config(config_path=None):
    """
    Load configuration from specified path or use default
    
    Args:
        config_path (str, optional): Path to configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    if not config_path:
        default_config_path = Path(__file__).parent / "config" / "default_config.json"
        config_path = default_config_path
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Validate the loaded configuration
        validate_config(config)
        return config
    except FileNotFoundError:
        logging.error(f"Configuration file not found: {config_path}")
        sys.exit(1)
    except json.JSONDecodeError:
        logging.error(f"Invalid JSON in configuration file: {config_path}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error loading configuration: {str(e)}")
        sys.exit(1)


def parse_arguments():
    """
    Parse command line arguments
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Warehouse Layout Generator")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--output", help="Output directory for generated files")
    parser.add_argument("--gui", action="store_true", help="Launch the graphical user interface")
    parser.add_argument("--optimize", action="store_true", help="Run space optimization algorithm")
    parser.add_argument("--verbose", "-v", action="count", default=0, help="Increase verbosity")
    parser.add_argument("--input", help="Input file with warehouse specifications")
    parser.add_argument("--export-format", choices=["dxf", "json", "csv"], default="dxf",
                        help="Export format for the generated layout")
    return parser.parse_args()


def main():
    """
    Main function to orchestrate the warehouse layout generation process
    """
    # Parse command line arguments
    args = parse_arguments()
    
    # Setup logging
    log_level = max(logging.WARNING - args.verbose * 10, logging.DEBUG)
    setup_logging(log_level)
    
    # Load configuration
    config = load_config(args.config)
    
    # Determine output directory
    output_dir = args.output or os.path.join(os.getcwd(), "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Launch GUI if requested
    if args.gui:
        logging.info("Launching graphical user interface")
        launch_gui(config)
        return
    
    # If no input file is provided and not in GUI mode, show help and exit
    if not args.input and not args.gui:
        logging.error("No input file provided. Use --input or --gui")
        return
    
    try:
        # Load warehouse specifications from input file
        if args.input:
            with open(args.input, 'r') as f:
                specs = json.load(f)
                
            # Create warehouse object
            warehouse = Warehouse.from_dict(specs)
            
            # Initialize layout engine
            layout_engine = LayoutEngine(warehouse, config)
            
            # Use AI-powered layout generator if optimization flag is set
            if args.optimize:
                logging.info("Running AI optimization algorithm")
                layout_generator = LayoutGenerator(config)
                optimized_layout = layout_generator.generate_optimized_layout(warehouse)
                layout_engine.apply_layout(optimized_layout)
            else:
                # Generate standard layout
                logging.info("Generating standard layout")
                layout_engine.generate_layout()
            
            # Export the layout in the requested format
            output_file = os.path.join(output_dir, f"warehouse_layout.{args.export_format}")
            if args.export_format == "dxf":
                export_to_dxf(layout_engine.get_layout(), output_file)
            elif args.export_format == "json":
                with open(output_file, 'w') as f:
                    json.dump(layout_engine.get_layout().to_dict(), f, indent=2)
            elif args.export_format == "csv":
                layout_engine.get_layout().export_to_csv(output_file)
                
            logging.info(f"Layout exported to {output_file}")
            
            # Process any CLI commands
            process_cli_commands(layout_engine, args)
            
    except Exception as e:
        logging.error(f"Error generating warehouse layout: {str(e)}")
        logging.debug("Exception details:", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()