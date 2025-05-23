"""
GUI implementation for Warehouse Layout Generator using tkinter.
Provides interface for configuring warehouse parameters and visualizing layouts.
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import json
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Add parent directory to path to allow imports from other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.layout_engine import LayoutEngine
from core.warehouse_elements import Warehouse, Rack, Aisle
from cad.export import export_to_dxf
from config import default_config
from utils.validators import validate_warehouse_parameters

class WarehouseLayoutGeneratorGUI:
    def __init__(self, root):
        """Initialize the GUI application"""
        self.root = root
        self.root.title("Warehouse Layout Generator")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)
        
        # Initialize instance variables
        self.layout_engine = LayoutEngine()
        self.current_layout = None
        self.current_config = {}
        self.load_default_config()
        
        # Create GUI components
        self.create_menu()
        self.create_main_layout()
        
        # Apply theme
        self.apply_theme()
        
    def load_default_config(self):
        """Load the default configuration"""
        try:
            # The actual implementation will load from the config module
            self.current_config = default_config.get_default_config()
        except:
            # Fallback default config if module import fails
            self.current_config = {
                "warehouse": {
                    "width": 100.0,
                    "length": 200.0,
                    "height": 10.0,
                    "dock_count": 5
                },
                "racks": {
                    "type": "standard",
                    "width": 2.5,
                    "depth": 1.2,
                    "height": 8.0,
                    "spacing": 0.15
                },
                "aisles": {
                    "main_width": 4.0,
                    "cross_width": 3.0,
                    "min_count": 2
                },
                "optimization": {
                    "method": "genetic",
                    "storage_priority": 0.7,
                    "accessibility_priority": 0.3,
                    "max_iterations": 100
                }
            }
    
    def create_menu(self):
        """Create the menu bar"""
        menubar = tk.Menu(self.root)
        
        # File menu
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="New Layout", command=self.new_layout)
        filemenu.add_command(label="Load Configuration", command=self.load_config)
        filemenu.add_command(label="Save Configuration", command=self.save_config)
        filemenu.add_separator()
        filemenu.add_command(label="Export to DXF", command=self.export_to_dxf_dialog)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=filemenu)
        
        # Edit menu
        editmenu = tk.Menu(menubar, tearoff=0)
        editmenu.add_command(label="Reset to Defaults", command=self.reset_to_defaults)
        menubar.add_cascade(label="Edit", menu=editmenu)
        
        # View menu
        viewmenu = tk.Menu(menubar, tearoff=0)
        viewmenu.add_command(label="Zoom In", command=self.zoom_in)
        viewmenu.add_command(label="Zoom Out", command=self.zoom_out)
        viewmenu.add_command(label="Fit to Screen", command=self.fit_to_screen)
        menubar.add_cascade(label="View", menu=viewmenu)
        
        # Help menu
        helpmenu = tk.Menu(menubar, tearoff=0)
        helpmenu.add_command(label="Documentation", command=self.show_documentation)
        helpmenu.add_command(label="About", command=self.show_about)
        menubar.add_cascade(label="Help", menu=helpmenu)
        
        self.root.config(menu=menubar)
    
    def create_main_layout(self):
        """Create the main layout of the GUI"""
        # Create main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create left panel (parameters)
        self.param_frame = ttk.LabelFrame(main_frame, text="Warehouse Parameters")
        self.param_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # Create notebook for categorizing parameters
        self.param_notebook = ttk.Notebook(self.param_frame)
        self.param_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create tabs for different parameter categories
        self.create_warehouse_tab()
        self.create_rack_tab()
        self.create_aisle_tab()
        self.create_optimization_tab()
        
        # Action buttons
        btn_frame = ttk.Frame(self.param_frame)
        btn_frame.pack(fill=tk.X, padx=5, pady=10)
        
        generate_btn = ttk.Button(btn_frame, text="Generate Layout", command=self.generate_layout)
        generate_btn.pack(side=tk.LEFT, padx=5)
        
        save_btn = ttk.Button(btn_frame, text="Save Layout", command=self.save_layout)
        save_btn.pack(side=tk.RIGHT, padx=5)
        
        # Create right panel (visualization)
        viz_frame = ttk.LabelFrame(main_frame, text="Layout Visualization")
        viz_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create figure for layout visualization
        self.fig = plt.Figure(figsize=(6, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_aspect('equal')
        self.canvas = FigureCanvasTkAgg(self.fig, master=viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def create_warehouse_tab(self):
        """Create the warehouse parameters tab"""
        warehouse_frame = ttk.Frame(self.param_notebook)
        self.param_notebook.add(warehouse_frame, text="Warehouse")
        
        # Warehouse dimensions
        ttk.Label(warehouse_frame, text="Width (m):").grid(row=0, column=0, sticky=tk.W, padx=10, pady=5)
        self.warehouse_width_var = tk.DoubleVar(value=self.current_config["warehouse"]["width"])
        ttk.Entry(warehouse_frame, textvariable=self.warehouse_width_var, width=10).grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(warehouse_frame, text="Length (m):").grid(row=1, column=0, sticky=tk.W, padx=10, pady=5)
        self.warehouse_length_var = tk.DoubleVar(value=self.current_config["warehouse"]["length"])
        ttk.Entry(warehouse_frame, textvariable=self.warehouse_length_var, width=10).grid(row=1, column=1, padx=5, pady=5)
        
        ttk.Label(warehouse_frame, text="Height (m):").grid(row=2, column=0, sticky=tk.W, padx=10, pady=5)
        self.warehouse_height_var = tk.DoubleVar(value=self.current_config["warehouse"]["height"])
        ttk.Entry(warehouse_frame, textvariable=self.warehouse_height_var, width=10).grid(row=2, column=1, padx=5, pady=5)
        
        ttk.Label(warehouse_frame, text="Loading Docks:").grid(row=3, column=0, sticky=tk.W, padx=10, pady=5)
        self.dock_count_var = tk.IntVar(value=self.current_config["warehouse"]["dock_count"])
        ttk.Entry(warehouse_frame, textvariable=self.dock_count_var, width=10).grid(row=3, column=1, padx=5, pady=5)
        
        # Optional: Add more warehouse-specific parameters
        
        # Add some padding at the bottom
        ttk.Frame(warehouse_frame).grid(row=10, column=0, pady=10)
    
    def create_rack_tab(self):
        """Create the rack parameters tab"""
        rack_frame = ttk.Frame(self.param_notebook)
        self.param_notebook.add(rack_frame, text="Racks")
        
        ttk.Label(rack_frame, text="Rack Type:").grid(row=0, column=0, sticky=tk.W, padx=10, pady=5)
        self.rack_type_var = tk.StringVar(value=self.current_config["racks"]["type"])
        rack_types = ["standard", "double-deep", "drive-in", "push-back"]
        ttk.Combobox(rack_frame, textvariable=self.rack_type_var, values=rack_types, width=15).grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(rack_frame, text="Width (m):").grid(row=1, column=0, sticky=tk.W, padx=10, pady=5)
        self.rack_width_var = tk.DoubleVar(value=self.current_config["racks"]["width"])
        ttk.Entry(rack_frame, textvariable=self.rack_width_var, width=10).grid(row=1, column=1, padx=5, pady=5)
        
        ttk.Label(rack_frame, text="Depth (m):").grid(row=2, column=0, sticky=tk.W, padx=10, pady=5)
        self.rack_depth_var = tk.DoubleVar(value=self.current_config["racks"]["depth"])
        ttk.Entry(rack_frame, textvariable=self.rack_depth_var, width=10).grid(row=2, column=1, padx=5, pady=5)
        
        ttk.Label(rack_frame, text="Height (m):").grid(row=3, column=0, sticky=tk.W, padx=10, pady=5)
        self.rack_height_var = tk.DoubleVar(value=self.current_config["racks"]["height"])
        ttk.Entry(rack_frame, textvariable=self.rack_height_var, width=10).grid(row=3, column=1, padx=5, pady=5)
        
        ttk.Label(rack_frame, text="Spacing (m):").grid(row=4, column=0, sticky=tk.W, padx=10, pady=5)
        self.rack_spacing_var = tk.DoubleVar(value=self.current_config["racks"]["spacing"])
        ttk.Entry(rack_frame, textvariable=self.rack_spacing_var, width=10).grid(row=4, column=1, padx=5, pady=5)
    
    def create_aisle_tab(self):
        """Create the aisle parameters tab"""
        aisle_frame = ttk.Frame(self.param_notebook)
        self.param_notebook.add(aisle_frame, text="Aisles")
        
        ttk.Label(aisle_frame, text="Main Aisle Width (m):").grid(row=0, column=0, sticky=tk.W, padx=10, pady=5)
        self.main_aisle_width_var = tk.DoubleVar(value=self.current_config["aisles"]["main_width"])
        ttk.Entry(aisle_frame, textvariable=self.main_aisle_width_var, width=10).grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(aisle_frame, text="Cross Aisle Width (m):").grid(row=1, column=0, sticky=tk.W, padx=10, pady=5)
        self.cross_aisle_width_var = tk.DoubleVar(value=self.current_config["aisles"]["cross_width"])
        ttk.Entry(aisle_frame, textvariable=self.cross_aisle_width_var, width=10).grid(row=1, column=1, padx=5, pady=5)
        
        ttk.Label(aisle_frame, text="Minimum Aisle Count:").grid(row=2, column=0, sticky=tk.W, padx=10, pady=5)
        self.min_aisle_count_var = tk.IntVar(value=self.current_config["aisles"]["min_count"])
        ttk.Entry(aisle_frame, textvariable=self.min_aisle_count_var, width=10).grid(row=2, column=1, padx=5, pady=5)
    
    def create_optimization_tab(self):
        """Create the optimization parameters tab"""
        opt_frame = ttk.Frame(self.param_notebook)
        self.param_notebook.add(opt_frame, text="Optimization")
        
        ttk.Label(opt_frame, text="Method:").grid(row=0, column=0, sticky=tk.W, padx=10, pady=5)
        self.opt_method_var = tk.StringVar(value=self.current_config["optimization"]["method"])
        opt_methods = ["genetic", "simulated-annealing", "particle-swarm", "manual"]
        ttk.Combobox(opt_frame, textvariable=self.opt_method_var, values=opt_methods, width=15).grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(opt_frame, text="Storage Priority:").grid(row=1, column=0, sticky=tk.W, padx=10, pady=5)
        self.storage_priority_var = tk.DoubleVar(value=self.current_config["optimization"]["storage_priority"])
        storage_scale = ttk.Scale(opt_frame, from_=0, to=1, orient=tk.HORIZONTAL, variable=self.storage_priority_var)
        storage_scale.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W+tk.E)
        
        ttk.Label(opt_frame, text="Accessibility Priority:").grid(row=2, column=0, sticky=tk.W, padx=10, pady=5)
        self.accessibility_priority_var = tk.DoubleVar(value=self.current_config["optimization"]["accessibility_priority"])
        access_scale = ttk.Scale(opt_frame, from_=0, to=1, orient=tk.HORIZONTAL, variable=self.accessibility_priority_var)
        access_scale.grid(row=2, column=1, padx=5, pady=5, sticky=tk.W+tk.E)
        
        ttk.Label(opt_frame, text="Max Iterations:").grid(row=3, column=0, sticky=tk.W, padx=10, pady=5)
        self.max_iterations_var = tk.IntVar(value=self.current_config["optimization"]["max_iterations"])
        ttk.Entry(opt_frame, textvariable=self.max_iterations_var, width=10).grid(row=3, column=1, padx=5, pady=5)
    
    def apply_theme(self):
        """Apply a consistent theme to the GUI"""
        style = ttk.Style()
        
        # Try to use a platform-specific theme if available
        try:
            if sys.platform.startswith('win'):
                style.theme_use('vista')
            elif sys.platform.startswith('darwin'):
                style.theme_use('aqua')
            else:
                style.theme_use('clam')
        except tk.TclError:
            # Fallback to a safe theme
            style.theme_use('clam')
        
        # Configure custom styles
        style.configure('TButton', font=('Arial', 10))
        style.configure('TLabel', font=('Arial', 10))
        style.configure('TLabelframe.Label', font=('Arial', 10, 'bold'))
    
    def gather_config_from_ui(self):
        """Gather configuration from the UI inputs"""
        config = {
            "warehouse": {
                "width": self.warehouse_width_var.get(),
                "length": self.warehouse_length_var.get(),
                "height": self.warehouse_height_var.get(),
                "dock_count": self.dock_count_var.get()
            },
            "racks": {
                "type": self.rack_type_var.get(),
                "width": self.rack_width_var.get(),
                "depth": self.rack_depth_var.get(),
                "height": self.rack_height_var.get(),
                "spacing": self.rack_spacing_var.get()
            },
            "aisles": {
                "main_width": self.main_aisle_width_var.get(),
                "cross_width": self.cross_aisle_width_var.get(),
                "min_count": self.min_aisle_count_var.get()
            },
            "optimization": {
                "method": self.opt_method_var.get(),
                "storage_priority": self.storage_priority_var.get(),
                "accessibility_priority": self.accessibility_priority_var.get(),
                "max_iterations": self.max_iterations_var.get()
            }
        }
        return config
    
    def update_ui_from_config(self, config):
        """Update the UI inputs from the configuration"""
        # Warehouse params
        self.warehouse_width_var.set(config["warehouse"]["width"])
        self.warehouse_length_var.set(config["warehouse"]["length"])
        self.warehouse_height_var.set(config["warehouse"]["height"])
        self.dock_count_var.set(config["warehouse"]["dock_count"])
        
        # Rack params
        self.rack_type_var.set(config["racks"]["type"])
        self.rack_width_var.set(config["racks"]["width"])
        self.rack_depth_var.set(config["racks"]["depth"])
        self.rack_height_var.set(config["racks"]["height"])
        self.rack_spacing_var.set(config["racks"]["spacing"])
        
        # Aisle params
        self.main_aisle_width_var.set(config["aisles"]["main_width"])
        self.cross_aisle_width_var.set(config["aisles"]["cross_width"])
        self.min_aisle_count_var.set(config["aisles"]["min_count"])
        
        # Optimization params
        self.opt_method_var.set(config["optimization"]["method"])
        self.storage_priority_var.set(config["optimization"]["storage_priority"])
        self.accessibility_priority_var.set(config["optimization"]["accessibility_priority"])
        self.max_iterations_var.set(config["optimization"]["max_iterations"])
    
    def generate_layout(self):
        """Generate a warehouse layout based on current parameters"""
        # Get current config from UI
        config = self.gather_config_from_ui()
        
        # Validate parameters
        try:
            valid = validate_warehouse_parameters(config)
            if not valid:
                messagebox.showerror("Validation Error", "Invalid warehouse parameters. Please check your inputs.")
                return
        except Exception as e:
            messagebox.showerror("Validation Error", f"Error validating parameters: {str(e)}")
            return
        
        # Update status
        self.status_var.set("Generating layout...")
        self.root.update_idletasks()
        
        # Generate layout in a separate thread to avoid freezing UI
        def generate_thread():
            try:
                # Create warehouse object
                warehouse = Warehouse(
                    width=config["warehouse"]["width"],
                    length=config["warehouse"]["length"],
                    height=config["warehouse"]["height"],
                    dock_count=config["warehouse"]["dock_count"]
                )
                
                # Generate layout using layout engine
                self.current_layout = self.layout_engine.generate_layout(
                    warehouse=warehouse,
                    rack_params=config["racks"],
                    aisle_params=config["aisles"],
                    optimization_params=config["optimization"]
                )
                
                # Update UI in the main thread
                self.root.after(0, self.update_visualization)
                self.root.after(0, lambda: self.status_var.set("Layout generated successfully"))
            except Exception as e:
                # Handle errors
                self.root.after(0, lambda: messagebox.showerror("Generation Error", f"Error generating layout: {str(e)}"))
                self.root.after(0, lambda: self.status_var.set("Error generating layout"))
        
        # Start generation thread
        threading.Thread(target=generate_thread).start()
    
    def update_visualization(self):
        """Update the visualization with the current layout"""
        if not self.current_layout:
            return
        
        # Clear previous plot
        self.ax.clear()
        
        # Set up the plot
        self.ax.set_aspect('equal')
        self.ax.set_xlabel('Width (m)')
        self.ax.set_ylabel('Length (m)')
        self.ax.set_title('Warehouse Layout')
        
        # Get warehouse dimensions
        width = self.warehouse_width_var.get()
        length = self.warehouse_length_var.get()
        
        # Draw warehouse outline
        self.ax.add_patch(plt.Rectangle((0, 0), width, length, fill=False, color='black', linewidth=2))
        
        # Draw racks - in a real implementation these would come from self.current_layout
        # This is a simplified visualization for demonstration
        if hasattr(self.current_layout, 'racks'):
            for rack in self.current_layout.racks:
                self.ax.add_patch(plt.Rectangle((rack.x, rack.y), rack.width, rack.depth, 
                                              fill=True, color='blue', alpha=0.5))
        else:
            # Demo visualization if no real layout data
            rack_width = self.rack_width_var.get()
            rack_depth = self.rack_depth_var.get()
            aisle_width = self.main_aisle_width_var.get()
            
            # Simple demo layout
            for i in range(3):  # 3 rows of racks
                y_pos = 10 + i * (2 * rack_depth + aisle_width)
                for j in range(5):  # 5 racks per row
                    x_pos = 5 + j * (rack_width + 1)
                    self.ax.add_patch(plt.Rectangle((x_pos, y_pos), rack_width, rack_depth, 
                                                  fill=True, color='blue', alpha=0.5))
        
        # Draw aisles, docks, etc. (simplified for demo)
        
        # Update the canvas
        self.canvas.draw()
    
    def new_layout(self):
        """Create a new layout"""
        if messagebox.askyesno("New Layout", "Clear current layout and start fresh?"):
            self.current_layout = None
            self.ax.clear()
            self.canvas.draw()
            self.status_var.set("Ready for new layout")
    
    def load_config(self):
        """Load configuration from a JSON file"""
        file_path = filedialog.askopenfilename(defaultextension=".json", 
                                             filetypes=[("JSON files", "*.json"), ("All files", "*.*")])
        if not file_path:
            return
        
        try:
            with open(file_path, 'r') as f:
                config = json.load(f)
            
            # Update current config
            self.current_config = config
            self.update_ui_from_config(config)
            self.status_var.set(f"Configuration loaded from {os.path.basename(file_path)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load configuration: {str(e)}")
    
    def save_config(self):
        """Save current configuration to a JSON file"""
        file_path = filedialog.asksaveasfilename(defaultextension=".json", 
                                              filetypes=[("JSON files", "*.json"), ("All files", "*.*")])
        if not file_path:
            return
        
        try:
            config = self.gather_config_from_ui()
            with open(file_path, 'w') as f:
                json.dump(config, f, indent=4)
            
            self.status_var.set(f"Configuration saved to {os.path.basename(file_path)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save configuration: {str(e)}")
    
    def save_layout(self):
        """Save current layout"""
        if not self.current_layout:
            messagebox.showwarning("Warning", "No layout to save. Generate a layout first.")
            return
        
        file_path = filedialog.asksaveasfilename(defaultextension=".wlg", 
                                             filetypes=[("Warehouse Layout files", "*.wlg"), ("All files", "*.*")])
        if not file_path:
            return
        
        try:
            # In a real implementation, this would serialize the layout object
            # For this demo, we'll just save the configuration used to generate it
            config = self.gather_config_from_ui()
            with open(file_path, 'w') as f:
                json.dump({
                    "config": config,
                    "layout_data": "serialized_layout_would_go_here"
                }, f, indent=4)
            
            self.status_var.set(f"Layout saved to {os.path.basename(file_path)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save layout: {str(e)}")
    
    def export_to_dxf_dialog(self):
        """Export the current layout to DXF file"""
        if not self.current_layout:
            messagebox.showwarning("Warning", "No layout to export. Generate a layout first.")
            return
        
        file_path = filedialog.asksaveasfilename(defaultextension=".dxf", 
                                              filetypes=[("DXF files", "*.dxf"), ("All files", "*.*")])
        if not file_path:
            return
        
        try:
            # Use the export module
            export_to_dxf(self.current_layout, file_path)
            self.status_var.set(f"Layout exported to {os.path.basename(file_path)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export layout: {str(e)}")
    
    def reset_to_defaults(self):
        """Reset parameters to default values"""
        if messagebox.askyesno("Reset", "Reset all parameters to default values?"):
            self.load_default_config()
            self.update_ui_from_config(self.current_config)
            self.status_var.set("Parameters reset to defaults")
    
    def zoom_in(self):
        """Zoom in the visualization"""
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        
        # Zoom in by 20%
        x_range = xlim[1] - xlim[0]
        y_range = ylim[1] - ylim[0]
        
        self.ax.set_xlim(xlim[0] + 0.1*x_range, xlim[1] - 0.1*x_range)
        self.ax.set_ylim(ylim[0] + 0.1*y_range, ylim[1] - 0.1*y_range)
        self.canvas.draw()
    
    def zoom_out(self):
        """Zoom out the visualization"""
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        
        # Zoom out by 20%
        x_range = xlim[1] - xlim[0]
        y_range = ylim[1] - ylim[0]
        
        self.ax.set_xlim(xlim[0] - 0.1*x_range, xlim[1] + 0.1*x_range)
        self.ax.set_ylim(ylim[0] - 0.1*y_range, ylim[1] + 0.1*y_range)
        self.canvas.draw()
    
    def fit_to_screen(self):
        """Fit the visualization to screen"""
        if not self.current_layout:
            # If no layout, just show the full warehouse dimensions
            self.ax.set_xlim(0, self.warehouse_width_var.get())
            self.ax.set_ylim(0, self.warehouse_length_var.get())
        else:
            # In a real implementation, this would get the bounds from the layout
            # For this demo, just show the full warehouse
            self.ax.set_xlim(0, self.warehouse_width_var.get())
            self.ax.set_ylim(0, self.warehouse_length_var.get())
        
        self.canvas.draw()
    
    def show_documentation(self):
        """Show documentation"""
        doc_text = """
        Warehouse Layout Generator Documentation
        
        This application allows you to design and optimize warehouse layouts.
        
        Basic workflow:
        1. Set warehouse dimensions and parameters
        2. Configure rack and aisle settings
        3. Set optimization priorities
        4. Generate the layout
        5. Export to DXF or save configuration for later use
        
        For more detailed documentation, please refer to the user manual.
        """
        
        doc_window = tk.Toplevel(self.root)
        doc_window.title("Documentation")
        doc_window.geometry("600x400")
        
        text_widget = tk.Text(doc_window, wrap=tk.WORD)
        text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        text_widget.insert(tk.END, doc_text)
        text_widget.config(state=tk.DISABLED)
        
        close_btn = ttk.Button(doc_window, text="Close", command=doc_window.destroy)
        close_btn.pack(pady=10)
    
    def show_about(self):
        """Show about dialog"""
        messagebox.showinfo("About", 
                          "Warehouse Layout Generator v1.0\n\n"
                          "A tool for optimizing warehouse layouts using AI and optimization algorithms.\n\n"
                          "© 2025 Your Company")
    
def main():
    """Main function to run the GUI application"""
    root = tk.Tk()
    app = WarehouseLayoutGeneratorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()