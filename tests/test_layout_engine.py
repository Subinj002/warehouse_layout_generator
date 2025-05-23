import unittest
import os
import sys
import json
from unittest.mock import MagicMock, patch

# Add the parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.layout_engine import LayoutEngine
from core.warehouse_elements import Rack, Aisle, Door, Zone
from core.constraints import ConstraintValidator
from core.optimization import LayoutOptimizer
from utils.geometry import Rectangle, Point


class TestLayoutEngine(unittest.TestCase):
    """Test suite for the Layout Engine functionality."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Mock configuration
        self.mock_config = {
            "warehouse": {
                "width": 100.0,
                "length": 200.0,
                "height": 10.0,
                "units": "meters"
            },
            "constraints": {
                "min_aisle_width": 3.0,
                "rack_clearance": 0.5,
                "door_clearance": 2.0,
                "max_rack_height": 8.0
            },
            "optimization": {
                "storage_capacity_weight": 0.7,
                "accessibility_weight": 0.3,
                "algorithm": "genetic"
            }
        }
        
        # Create test instance with mock objects
        self.constraint_validator = MagicMock(spec=ConstraintValidator)
        self.layout_optimizer = MagicMock(spec=LayoutOptimizer)
        
        # Create the layout engine with mocked dependencies
        self.layout_engine = LayoutEngine(
            config=self.mock_config,
            constraint_validator=self.constraint_validator,
            layout_optimizer=self.layout_optimizer
        )

    def test_initialization(self):
        """Test the layout engine initializes correctly with given config."""
        self.assertEqual(self.layout_engine.width, 100.0)
        self.assertEqual(self.layout_engine.length, 200.0)
        self.assertEqual(self.layout_engine.height, 10.0)
        self.assertEqual(self.layout_engine.units, "meters")
        self.assertEqual(self.layout_engine.elements, [])
        
    def test_add_element(self):
        """Test adding warehouse elements to the layout."""
        # Create a mock rack element
        rack = Rack(
            id="rack1",
            position=Point(10, 10),
            width=5.0,
            length=20.0,
            height=8.0,
            shelves=4
        )
        
        # Add the rack to the layout
        self.layout_engine.add_element(rack)
        
        # Verify the rack was added
        self.assertEqual(len(self.layout_engine.elements), 1)
        self.assertEqual(self.layout_engine.elements[0], rack)
        
        # Test adding multiple elements
        aisle = Aisle(
            id="aisle1",
            start_point=Point(20, 10),
            end_point=Point(20, 50),
            width=3.0
        )
        
        self.layout_engine.add_element(aisle)
        self.assertEqual(len(self.layout_engine.elements), 2)
        self.assertEqual(self.layout_engine.elements[1], aisle)
    
    def test_remove_element(self):
        """Test removing elements from the layout."""
        # Add elements
        rack1 = Rack(id="rack1", position=Point(10, 10), width=5.0, length=20.0, height=8.0, shelves=4)
        rack2 = Rack(id="rack2", position=Point(20, 10), width=5.0, length=20.0, height=8.0, shelves=4)
        
        self.layout_engine.add_element(rack1)
        self.layout_engine.add_element(rack2)
        
        # Remove one element
        self.layout_engine.remove_element("rack1")
        
        # Verify only rack2 remains
        self.assertEqual(len(self.layout_engine.elements), 1)
        self.assertEqual(self.layout_engine.elements[0].id, "rack2")
        
        # Test removing non-existent element
        with self.assertRaises(ValueError):
            self.layout_engine.remove_element("non_existent")
    
    def test_get_element_by_id(self):
        """Test retrieving specific elements by ID."""
        rack = Rack(id="rack1", position=Point(10, 10), width=5.0, length=20.0, height=8.0, shelves=4)
        door = Door(id="door1", position=Point(0, 50), width=4.0, is_emergency=False)
        
        self.layout_engine.add_element(rack)
        self.layout_engine.add_element(door)
        
        # Retrieve and verify
        retrieved_rack = self.layout_engine.get_element_by_id("rack1")
        self.assertEqual(retrieved_rack, rack)
        
        # Test non-existent element
        self.assertIsNone(self.layout_engine.get_element_by_id("non_existent"))
    
    def test_get_elements_by_type(self):
        """Test filtering elements by their type."""
        rack1 = Rack(id="rack1", position=Point(10, 10), width=5.0, length=20.0, height=8.0, shelves=4)
        rack2 = Rack(id="rack2", position=Point(20, 10), width=5.0, length=20.0, height=8.0, shelves=4)
        door = Door(id="door1", position=Point(0, 50), width=4.0, is_emergency=False)
        
        self.layout_engine.add_element(rack1)
        self.layout_engine.add_element(rack2)
        self.layout_engine.add_element(door)
        
        # Get all racks
        racks = self.layout_engine.get_elements_by_type(Rack)
        self.assertEqual(len(racks), 2)
        self.assertIn(rack1, racks)
        self.assertIn(rack2, racks)
        
        # Get all doors
        doors = self.layout_engine.get_elements_by_type(Door)
        self.assertEqual(len(doors), 1)
        self.assertIn(door, doors)
        
        # Get non-existent type
        zones = self.layout_engine.get_elements_by_type(Zone)
        self.assertEqual(len(zones), 0)
    
    def test_validate_layout(self):
        """Test layout validation using the constraint validator."""
        # Setup mock validator to return True
        self.constraint_validator.validate.return_value = True
        
        # Add some elements
        rack1 = Rack(id="rack1", position=Point(10, 10), width=5.0, length=20.0, height=8.0, shelves=4)
        rack2 = Rack(id="rack2", position=Point(20, 10), width=5.0, length=20.0, height=8.0, shelves=4)
        self.layout_engine.add_element(rack1)
        self.layout_engine.add_element(rack2)
        
        # Validate the layout
        validation_result = self.layout_engine.validate_layout()
        
        # Verify validator was called with the elements
        self.constraint_validator.validate.assert_called_once_with(
            self.layout_engine.elements, 
            self.mock_config["constraints"]
        )
        
        # Check the result
        self.assertTrue(validation_result)
    
    def test_optimize_layout(self):
        """Test layout optimization."""
        # Setup mock optimizer to return a new list of elements
        optimized_rack = Rack(id="optimized_rack", position=Point(15, 15), width=5.0, length=20.0, height=8.0, shelves=4)
        self.layout_optimizer.optimize.return_value = [optimized_rack]
        
        # Add some initial elements
        rack = Rack(id="rack1", position=Point(10, 10), width=5.0, length=20.0, height=8.0, shelves=4)
        self.layout_engine.add_element(rack)
        
        # Optimize the layout
        self.layout_engine.optimize_layout()
        
        # Verify optimizer was called with the right parameters
        self.layout_optimizer.optimize.assert_called_once_with(
            self.layout_engine.elements,
            self.layout_engine.width,
            self.layout_engine.length,
            self.mock_config["optimization"],
            self.mock_config["constraints"]
        )
        
        # Check the elements were updated
        self.assertEqual(len(self.layout_engine.elements), 1)
        self.assertEqual(self.layout_engine.elements[0], optimized_rack)
    
    @patch('builtins.open', new_callable=unittest.mock.mock_open)
    def test_save_layout(self, mock_file):
        """Test saving layout to a file."""
        # Add elements
        rack = Rack(id="rack1", position=Point(10, 10), width=5.0, length=20.0, height=8.0, shelves=4)
        self.layout_engine.add_element(rack)
        
        # Mock the element serialization method
        rack.to_dict = MagicMock(return_value={
            "id": "rack1",
            "type": "rack",
            "position": {"x": 10, "y": 10},
            "width": 5.0,
            "length": 20.0,
            "height": 8.0,
            "shelves": 4
        })
        
        # Save the layout
        self.layout_engine.save_layout("test_layout.json")
        
        # Verify file was opened for writing
        mock_file.assert_called_once_with("test_layout.json", "w")
        
        # Verify data was written to the file
        expected_data = {
            "warehouse": {
                "width": 100.0,
                "length": 200.0,
                "height": 10.0,
                "units": "meters"
            },
            "elements": [rack.to_dict()]
        }
        mock_file().write.assert_called_once_with(json.dumps(expected_data, indent=2))
    
    @patch('builtins.open', new_callable=unittest.mock.mock_open, read_data='{"warehouse": {"width": 80, "length": 150, "height": 12, "units": "meters"}, "elements": [{"id": "loaded_rack", "type": "rack", "position": {"x": 30, "y": 30}, "width": 5.0, "length": 20.0, "height": 8.0, "shelves": 4}]}')
    @patch('core.warehouse_elements.Rack.from_dict')
    def test_load_layout(self, mock_from_dict, mock_file):
        """Test loading layout from a file."""
        # Setup mock for element creation from dict
        loaded_rack = Rack(id="loaded_rack", position=Point(30, 30), width=5.0, length=20.0, height=8.0, shelves=4)
        mock_from_dict.return_value = loaded_rack
        
        # Load the layout
        self.layout_engine.load_layout("test_layout.json")
        
        # Verify file was opened for reading
        mock_file.assert_called_once_with("test_layout.json", "r")
        
        # Verify warehouse dimensions were updated
        self.assertEqual(self.layout_engine.width, 80)
        self.assertEqual(self.layout_engine.length, 150)
        self.assertEqual(self.layout_engine.height, 12)
        
        # Verify elements were loaded
        self.assertEqual(len(self.layout_engine.elements), 1)
        self.assertEqual(self.layout_engine.elements[0], loaded_rack)
    
    def test_calculate_storage_capacity(self):
        """Test calculating the total storage capacity of the layout."""
        # Mock racks with storage capacities
        rack1 = MagicMock(spec=Rack)
        rack1.calculate_capacity.return_value = 100
        rack1.__class__ = Rack
        
        rack2 = MagicMock(spec=Rack)
        rack2.calculate_capacity.return_value = 150
        rack2.__class__ = Rack
        
        # Add non-storage elements too
        door = Door(id="door1", position=Point(0, 50), width=4.0, is_emergency=False)
        
        # Add all elements
        self.layout_engine.elements = [rack1, rack2, door]
        
        # Calculate capacity
        capacity = self.layout_engine.calculate_storage_capacity()
        
        # Verify calculation
        self.assertEqual(capacity, 250)  # 100 + 150
        rack1.calculate_capacity.assert_called_once()
        rack2.calculate_capacity.assert_called_once()
    
    def test_calculate_floor_space_utilization(self):
        """Test calculating floor space utilization percentage."""
        # Add elements with floor space
        rack = Rack(id="rack1", position=Point(10, 10), width=10.0, length=20.0, height=8.0, shelves=4)
        aisle = Aisle(id="aisle1", start_point=Point(20, 10), end_point=Point(20, 50), width=5.0)
        
        # Override the footprint calculations for testing
        rack.calculate_footprint = MagicMock(return_value=200.0)  # 10 * 20
        aisle.calculate_footprint = MagicMock(return_value=200.0)  # 5 * 40
        
        # Add elements
        self.layout_engine.elements = [rack, aisle]
        
        # Calculate utilization (400 / 20000 = 0.02 = 2%)
        utilization = self.layout_engine.calculate_floor_space_utilization()
        
        # Verify calculation
        self.assertEqual(utilization, 0.02)  # (200 + 200) / (100 * 200)
        rack.calculate_footprint.assert_called_once()
        aisle.calculate_footprint.assert_called_once()
    
    def test_find_collisions(self):
        """Test detecting collisions between elements."""
        # Create elements with mock collision detection
        rack1 = MagicMock()
        rack1.id = "rack1"
        rack1.collides_with.return_value = False
        
        rack2 = MagicMock()
        rack2.id = "rack2"
        rack2.collides_with.side_effect = lambda x: x.id == "rack3"  # Collides with rack3
        
        rack3 = MagicMock()
        rack3.id = "rack3"
        rack3.collides_with.side_effect = lambda x: x.id == "rack2"  # Collides with rack2
        
        # Add elements
        self.layout_engine.elements = [rack1, rack2, rack3]
        
        # Find collisions
        collisions = self.layout_engine.find_collisions()
        
        # Verify collision detection
        self.assertEqual(len(collisions), 1)
        self.assertIn(("rack2", "rack3"), collisions)
        
        # Verify collision method calls
        rack1.collides_with.assert_any_call(rack2)
        rack1.collides_with.assert_any_call(rack3)
        rack2.collides_with.assert_any_call(rack1)
        rack2.collides_with.assert_any_call(rack3)
        rack3.collides_with.assert_any_call(rack1)
        rack3.collides_with.assert_any_call(rack2)
    
    def test_generate_default_layout(self):
        """Test generating a default layout based on warehouse dimensions."""
        # Mock the optimizer to return a predefined layout
        default_rack = Rack(id="default_rack", position=Point(5, 5), width=5.0, length=20.0, height=8.0, shelves=4)
        default_aisle = Aisle(id="default_aisle", start_point=Point(15, 0), end_point=Point(15, 200), width=3.0)
        
        self.layout_optimizer.generate_default_layout.return_value = [default_rack, default_aisle]
        
        # Generate default layout
        self.layout_engine.generate_default_layout()
        
        # Verify optimizer was called
        self.layout_optimizer.generate_default_layout.assert_called_once_with(
            100.0, 200.0, 10.0, self.mock_config["constraints"]
        )
        
        # Check elements were added
        self.assertEqual(len(self.layout_engine.elements), 2)
        self.assertIn(default_rack, self.layout_engine.elements)
        self.assertIn(default_aisle, self.layout_engine.elements)


if __name__ == "__main__":
    unittest.main()