import unittest
import os
import tempfile
from unittest.mock import patch, MagicMock

import ezdxf

from cad.ezdxf_interface import DXFExporter
from cad.drawing_utils import draw_rack, draw_aisle, draw_loading_dock
from cad.export import export_layout_to_dxf
from core.warehouse_elements import Rack, Aisle, LoadingDock, WarehouseLayout


class TestDXFExporter(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for test files
        self.test_dir = tempfile.TemporaryDirectory()
        self.test_file_path = os.path.join(self.test_dir.name, "test_layout.dxf")
        
        # Initialize exporter with test file
        self.exporter = DXFExporter(self.test_file_path)
    
    def tearDown(self):
        # Clean up temporary directory
        self.test_dir.cleanup()
    
    def test_init_creates_valid_dxf_doc(self):
        """Test that the exporter initializes with a valid DXF document"""
        self.assertIsNotNone(self.exporter.doc)
        self.assertIsNotNone(self.exporter.modelspace)
    
    def test_save_file(self):
        """Test that the file is saved correctly"""
        self.exporter.save()
        self.assertTrue(os.path.exists(self.test_file_path))
        
        # Verify it's a valid DXF file by loading it back
        doc = ezdxf.readfile(self.test_file_path)
        self.assertIsNotNone(doc)
    
    def test_add_layer(self):
        """Test adding a layer to the document"""
        layer_name = "TEST_LAYER"
        color = 1  # red
        
        self.exporter.add_layer(layer_name, color)
        
        # Check if layer was added
        self.assertIn(layer_name, self.exporter.doc.layers)
        layer = self.exporter.doc.layers.get(layer_name)
        self.assertEqual(layer.color, color)
    
    def test_draw_line(self):
        """Test drawing a line in the document"""
        start = (0, 0)
        end = (100, 100)
        layer = "TEST_LINE_LAYER"
        
        # Add layer first
        self.exporter.add_layer(layer, 2)
        
        # Draw line
        line = self.exporter.draw_line(start, end, layer)
        
        # Check line properties
        self.assertEqual(line.dxf.layer, layer)
        self.assertEqual(line.dxf.start, start)
        self.assertEqual(line.dxf.end, end)
    
    def test_draw_rectangle(self):
        """Test drawing a rectangle in the document"""
        origin = (0, 0)
        width = 100
        height = 50
        layer = "TEST_RECT_LAYER"
        
        # Add layer
        self.exporter.add_layer(layer, 3)
        
        # Draw rectangle
        lines = self.exporter.draw_rectangle(origin, width, height, layer)
        
        # Should return 4 lines
        self.assertEqual(len(lines), 4)
        
        # Verify all lines have the right layer
        for line in lines:
            self.assertEqual(line.dxf.layer, layer)


class TestDrawingUtils(unittest.TestCase):
    def setUp(self):
        # Create a mock exporter
        self.mock_exporter = MagicMock()
        
        # Sample elements for drawing
        self.rack = Rack(
            x=10, y=20,
            width=100, depth=40, height=200,
            num_shelves=5, capacity=1000, rack_type="PALLET"
        )
        
        self.aisle = Aisle(
            x=150, y=20,
            width=20, length=200
        )
        
        self.loading_dock = LoadingDock(
            x=300, y=50,
            width=100, depth=80,
            dock_type="SHIPPING"
        )
    
    def test_draw_rack(self):
        """Test drawing a rack element"""
        # Call the function with our mock
        draw_rack(self.mock_exporter, self.rack)
        
        # Verify exporter's draw_rectangle was called with correct parameters
        self.mock_exporter.draw_rectangle.assert_called_with(
            (self.rack.x, self.rack.y),
            self.rack.width,
            self.rack.depth,
            "RACKS"
        )
        
        # Verify add_text was called with rack_type
        self.mock_exporter.add_text.assert_called()
    
    def test_draw_aisle(self):
        """Test drawing an aisle element"""
        draw_aisle(self.mock_exporter, self.aisle)
        
        # Verify exporter's draw_rectangle was called with correct parameters
        self.mock_exporter.draw_rectangle.assert_called_with(
            (self.aisle.x, self.aisle.y),
            self.aisle.width,
            self.aisle.length,
            "AISLES"
        )
    
    def test_draw_loading_dock(self):
        """Test drawing a loading dock element"""
        draw_loading_dock(self.mock_exporter, self.loading_dock)
        
        # Verify exporter's draw_rectangle was called with correct parameters
        self.mock_exporter.draw_rectangle.assert_called_with(
            (self.loading_dock.x, self.loading_dock.y),
            self.loading_dock.width, 
            self.loading_dock.depth,
            "LOADING_DOCKS"
        )
        
        # Verify add_text was called with dock_type
        self.mock_exporter.add_text.assert_called()


class TestExportFunctions(unittest.TestCase):
    @patch('cad.export.DXFExporter')
    @patch('cad.export.draw_rack')
    @patch('cad.export.draw_aisle')
    @patch('cad.export.draw_loading_dock')
    def test_export_layout_to_dxf(self, mock_draw_dock, mock_draw_aisle, mock_draw_rack, mock_exporter_class):
        """Test the layout export function with all warehouse elements"""
        # Setup mock exporter instance
        mock_exporter = MagicMock()
        mock_exporter_class.return_value = mock_exporter
        
        # Create a warehouse layout with some elements
        layout = WarehouseLayout(width=1000, height=800)
        
        # Add elements to layout
        rack1 = Rack(x=10, y=20, width=100, depth=40, height=200, 
                    num_shelves=5, capacity=1000, rack_type="PALLET")
        rack2 = Rack(x=10, y=100, width=100, depth=40, height=200, 
                    num_shelves=5, capacity=1000, rack_type="SHELF")
        
        aisle1 = Aisle(x=150, y=20, width=20, length=200)
        
        dock1 = LoadingDock(x=300, y=50, width=100, depth=80, dock_type="SHIPPING")
        
        # Add elements to layout
        layout.add_element(rack1)
        layout.add_element(rack2)
        layout.add_element(aisle1)
        layout.add_element(dock1)
        
        # Call export function
        filepath = "/fake/path/test.dxf"
        export_layout_to_dxf(layout, filepath)
        
        # Verify exporter was created with the right path
        mock_exporter_class.assert_called_with(filepath)
        
        # Verify layers were added
        self.assertTrue(mock_exporter.add_layer.called)
        
        # Verify draw functions were called for each element
        self.assertEqual(mock_draw_rack.call_count, 2)
        self.assertEqual(mock_draw_aisle.call_count, 1)
        self.assertEqual(mock_draw_dock.call_count, 1)
        
        # Verify save was called
        mock_exporter.save.assert_called_once()


if __name__ == '__main__':
    unittest.main()