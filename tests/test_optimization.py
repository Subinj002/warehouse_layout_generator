import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from core.optimization import (
    OptimizationEngine,
    SpaceOptimizer,
    LayoutOptimizer,
    ConstraintValidator
)
from core.warehouse_elements import Rack, Aisle, PickingStation, DockDoor
from core.constraints import WarehouseConstraints


class TestOptimizationEngine(unittest.TestCase):
    """Test suite for the optimization engine base class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.optimization_engine = OptimizationEngine()
    
    def test_initialization(self):
        """Test if the optimization engine initializes properly."""
        self.assertIsNotNone(self.optimization_engine)
    
    def test_abstract_methods(self):
        """Test that abstract methods raise NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            self.optimization_engine.optimize()
    
    def test_register_objective_function(self):
        """Test registering an objective function."""
        def dummy_objective(x):
            return np.sum(x)
        
        self.optimization_engine.register_objective_function(dummy_objective)
        self.assertEqual(self.optimization_engine.objective_function, dummy_objective)


class TestSpaceOptimizer(unittest.TestCase):
    """Test suite for the space optimizer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.constraints = WarehouseConstraints()
        self.constraints.set_constraint("min_aisle_width", 2.0)
        self.constraints.set_constraint("min_distance_to_wall", 0.5)
        
        self.space_optimizer = SpaceOptimizer(constraints=self.constraints)
        
        # Mock warehouse elements
        self.racks = [
            Rack(id=1, x=10, y=10, width=2, length=20, height=5),
            Rack(id=2, x=15, y=10, width=2, length=20, height=5)
        ]
        
        self.aisles = [
            Aisle(id=1, x=13, y=10, width=2, length=20)
        ]
    
    def test_initialization(self):
        """Test if the space optimizer initializes properly."""
        self.assertIsNotNone(self.space_optimizer)
        self.assertEqual(self.space_optimizer.constraints, self.constraints)
    
    @patch('core.optimization.SpaceOptimizer._calculate_space_utilization')
    def test_optimize_space_utilization(self, mock_calculate):
        """Test the space utilization optimization."""
        mock_calculate.return_value = 0.85
        
        warehouse_elements = {
            'racks': self.racks,
            'aisles': self.aisles,
            'picking_stations': [],
            'dock_doors': []
        }
        
        result = self.space_optimizer.optimize(warehouse_elements, max_iterations=10)
        
        self.assertGreaterEqual(result['space_utilization'], 0.0)
        self.assertLessEqual(result['space_utilization'], 1.0)
        self.assertIn('optimized_elements', result)
    
    def test_validate_layout(self):
        """Test the layout validation method."""
        warehouse_elements = {
            'racks': self.racks,
            'aisles': self.aisles,
            'picking_stations': [],
            'dock_doors': []
        }
        
        validation_result = self.space_optimizer.validate_layout(warehouse_elements)
        self.assertIsInstance(validation_result, dict)
        self.assertIn('valid', validation_result)
        self.assertIn('violations', validation_result)


class TestLayoutOptimizer(unittest.TestCase):
    """Test suite for the layout optimizer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.constraints = WarehouseConstraints()
        self.constraints.set_constraint("min_aisle_width", 2.0)
        self.constraints.set_constraint("min_distance_between_racks", 0.5)
        
        self.layout_optimizer = LayoutOptimizer(constraints=self.constraints)
        
        # Mock warehouse elements
        self.warehouse_dimensions = (100, 50)  # width, length
        
        self.racks = [
            Rack(id=1, x=10, y=10, width=2, length=20, height=5),
            Rack(id=2, x=15, y=10, width=2, length=20, height=5)
        ]
        
        self.aisles = [
            Aisle(id=1, x=13, y=10, width=2, length=20)
        ]
        
        self.picking_stations = [
            PickingStation(id=1, x=5, y=5, width=3, length=3)
        ]
        
        self.dock_doors = [
            DockDoor(id=1, x=0, y=25, width=4, length=1, orientation='W')
        ]
    
    def test_initialization(self):
        """Test if the layout optimizer initializes properly."""
        self.assertIsNotNone(self.layout_optimizer)
        self.assertEqual(self.layout_optimizer.constraints, self.constraints)
    
    @patch('core.optimization.LayoutOptimizer._calculate_travel_distances')
    def test_optimize_travel_distance(self, mock_travel):
        """Test the travel distance optimization."""
        mock_travel.return_value = {
            'avg_distance': 25.5,
            'max_distance': 50.2,
            'distances': [(1, 1, 24.5), (1, 2, 26.5)]
        }
        
        warehouse_elements = {
            'dimensions': self.warehouse_dimensions,
            'racks': self.racks,
            'aisles': self.aisles,
            'picking_stations': self.picking_stations,
            'dock_doors': self.dock_doors
        }
        
        result = self.layout_optimizer.optimize(
            warehouse_elements, 
            optimization_goal='minimize_travel_distance',
            max_iterations=10
        )
        
        self.assertIn('avg_travel_distance', result)
        self.assertIn('optimized_elements', result)
    
    @patch('core.optimization.LayoutOptimizer._calculate_picking_efficiency')
    def test_optimize_picking_efficiency(self, mock_efficiency):
        """Test the picking efficiency optimization."""
        mock_efficiency.return_value = 0.78
        
        warehouse_elements = {
            'dimensions': self.warehouse_dimensions,
            'racks': self.racks,
            'aisles': self.aisles,
            'picking_stations': self.picking_stations,
            'dock_doors': self.dock_doors
        }
        
        result = self.layout_optimizer.optimize(
            warehouse_elements, 
            optimization_goal='maximize_picking_efficiency',
            max_iterations=10
        )
        
        self.assertIn('picking_efficiency', result)
        self.assertIn('optimized_elements', result)


class TestConstraintValidator(unittest.TestCase):
    """Test suite for the constraint validator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.constraints = WarehouseConstraints()
        self.constraints.set_constraint("min_aisle_width", 2.0)
        self.constraints.set_constraint("min_distance_to_wall", 0.5)
        self.constraints.set_constraint("min_headroom", 0.5)
        
        self.validator = ConstraintValidator(constraints=self.constraints)
        
        # Mock warehouse elements
        self.warehouse_dimensions = (100, 50)  # width, length
        
        self.valid_rack = Rack(id=1, x=10, y=10, width=2, length=20, height=5)
        self.invalid_rack = Rack(id=2, x=-1, y=10, width=2, length=20, height=5)  # Outside boundary
        
        self.valid_aisle = Aisle(id=1, x=13, y=10, width=2, length=20)
        self.invalid_aisle = Aisle(id=2, x=13, y=10, width=1, length=20)  # Too narrow
        
        self.valid_picking_station = PickingStation(id=1, x=5, y=5, width=3, length=3)
        self.invalid_picking_station = PickingStation(id=2, x=98, y=48, width=3, length=3)  # Outside boundary
    
    def test_initialization(self):
        """Test if the constraint validator initializes properly."""
        self.assertIsNotNone(self.validator)
        self.assertEqual(self.validator.constraints, self.constraints)
    
    def test_validate_rack_position_valid(self):
        """Test validation of a valid rack position."""
        result = self.validator.validate_rack_position(
            self.valid_rack, 
            self.warehouse_dimensions, 
            [self.valid_rack]
        )
        self.assertTrue(result['valid'])
        self.assertEqual(len(result['violations']), 0)
    
    def test_validate_rack_position_invalid(self):
        """Test validation of an invalid rack position."""
        result = self.validator.validate_rack_position(
            self.invalid_rack, 
            self.warehouse_dimensions, 
            [self.invalid_rack]
        )
        self.assertFalse(result['valid'])
        self.assertGreater(len(result['violations']), 0)
    
    def test_validate_aisle_width_valid(self):
        """Test validation of a valid aisle width."""
        result = self.validator.validate_aisle_width(self.valid_aisle)
        self.assertTrue(result['valid'])
        self.assertEqual(len(result['violations']), 0)
    
    def test_validate_aisle_width_invalid(self):
        """Test validation of an invalid aisle width."""
        result = self.validator.validate_aisle_width(self.invalid_aisle)
        self.assertFalse(result['valid'])
        self.assertGreater(len(result['violations']), 0)
    
    def test_validate_complete_layout(self):
        """Test validation of a complete warehouse layout."""
        warehouse_elements = {
            'dimensions': self.warehouse_dimensions,
            'racks': [self.valid_rack],
            'aisles': [self.valid_aisle],
            'picking_stations': [self.valid_picking_station],
            'dock_doors': []
        }
        
        result = self.validator.validate_layout(warehouse_elements)
        self.assertIsInstance(result, dict)
        self.assertIn('valid', result)
        self.assertIn('violations', result)
        
        # Test with invalid elements
        invalid_warehouse_elements = {
            'dimensions': self.warehouse_dimensions,
            'racks': [self.invalid_rack],
            'aisles': [self.invalid_aisle],
            'picking_stations': [self.invalid_picking_station],
            'dock_doors': []
        }
        
        invalid_result = self.validator.validate_layout(invalid_warehouse_elements)
        self.assertFalse(invalid_result['valid'])
        self.assertGreater(len(invalid_result['violations']), 0)


if __name__ == '__main__':
    unittest.main()