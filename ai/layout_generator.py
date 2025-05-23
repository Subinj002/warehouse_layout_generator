"""
Warehouse Layout Generator - AI Layout Generator Module ==

This module contains the AI-based layout generation functionality for the warehouse layout generator.
It provides classes and methods for generating optimal warehouse layouts using different AI techniques
including rule-based systems, genetic algorithms, and reinforcement learning approaches.
"""

import random
import logging
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

from core.layout_engine import LayoutEngine
from core.warehouse_elements import (
    Rack, Aisle, Shelf, Dock, StorageZone, WorkArea, 
    PickingStation, WarehouseElement
)
from core.constraints import ConstraintManager
from utils.geometry import Rectangle, Point
from ai.space_optimizer import SpaceOptimizer
from ai.models import LayoutScoreModel

logger = logging.getLogger(__name__)

class LayoutGenerator:
    """Base class for AI-based layout generation strategies."""
    
    def __init__(self, 
                 layout_engine: LayoutEngine, 
                 constraint_manager: ConstraintManager,
                 config: Dict[str, Any]):
        """
        Initialize the layout generator.
        
        Args:
            layout_engine: The layout engine instance to use for layout manipulation
            constraint_manager: The constraint manager to validate layouts
            config: Configuration parameters for the layout generator
        """
        self.layout_engine = layout_engine
        self.constraint_manager = constraint_manager
        self.config = config
        self.score_model = LayoutScoreModel()
        self.space_optimizer = SpaceOptimizer(self.layout_engine, self.config)
        
    def generate_layout(self, warehouse_dimensions: Rectangle, 
                       requirements: Dict[str, Any]) -> Dict[str, List[WarehouseElement]]:
        """
        Generate a warehouse layout based on the given dimensions and requirements.
        
        Args:
            warehouse_dimensions: The dimensions of the warehouse
            requirements: The requirements for the warehouse layout
            
        Returns:
            A dictionary containing lists of warehouse elements by type
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def evaluate_layout(self, layout: Dict[str, List[WarehouseElement]]) -> float:
        """
        Evaluate the quality of a layout.
        
        Args:
            layout: The layout to evaluate
            
        Returns:
            A score representing the quality of the layout (higher is better)
        """
        # Use the layout score model to evaluate the layout
        return self.score_model.score_layout(layout, self.layout_engine)
    
    def optimize_layout(self, 
                       layout: Dict[str, List[WarehouseElement]], 
                       iterations: int = 100) -> Dict[str, List[WarehouseElement]]:
        """
        Optimize an existing layout to improve its score.
        
        Args:
            layout: The layout to optimize
            iterations: The number of optimization iterations to perform
            
        Returns:
            The optimized layout
        """
        # Use the space optimizer to optimize the layout
        return self.space_optimizer.optimize(layout, iterations)


class RuleBasedLayoutGenerator(LayoutGenerator):
    """Generate warehouse layouts using rule-based heuristics."""
    
    def generate_layout(self, warehouse_dimensions: Rectangle, 
                       requirements: Dict[str, Any]) -> Dict[str, List[WarehouseElement]]:
        """
        Generate a warehouse layout using rule-based heuristics.
        
        Args:
            warehouse_dimensions: The dimensions of the warehouse
            requirements: The requirements for the warehouse layout
            
        Returns:
            A dictionary containing lists of warehouse elements by type
        """
        logger.info("Generating rule-based layout")
        
        # Initialize empty layout
        layout = {
            'racks': [],
            'aisles': [],
            'docks': [],
            'storage_zones': [],
            'work_areas': [],
            'picking_stations': [],
            'shelves': []
        }
        
        # Extract requirements
        num_racks = requirements.get('num_racks', 10)
        num_aisles = requirements.get('num_aisles', 5)
        num_docks = requirements.get('num_docks', 2)
        
        # Place docks on the edges
        self._place_docks(layout, warehouse_dimensions, num_docks)
        
        # Place aisles
        self._place_aisles(layout, warehouse_dimensions, num_aisles)
        
        # Place racks around aisles
        self._place_racks(layout, warehouse_dimensions, num_racks)
        
        # Place work areas
        if 'work_areas' in requirements:
            self._place_work_areas(layout, warehouse_dimensions, requirements['work_areas'])
        
        # Place picking stations
        if 'picking_stations' in requirements:
            self._place_picking_stations(layout, warehouse_dimensions, requirements['picking_stations'])
        
        # Validate and fix constraints
        valid, violations = self.constraint_manager.validate_layout(layout)
        if not valid:
            logger.warning(f"Layout has {len(violations)} constraint violations")
            layout = self._fix_constraint_violations(layout, violations)
        
        return layout
    
    def _place_docks(self, 
                    layout: Dict[str, List[WarehouseElement]], 
                    warehouse_dimensions: Rectangle, 
                    num_docks: int):
        """Place loading docks along the warehouse perimeter."""
        dock_width = self.config.get('dock_width', 4.0)
        dock_depth = self.config.get('dock_depth', 3.0)
        
        # Place docks on the south wall
        available_width = warehouse_dimensions.width
        dock_spacing = available_width / (num_docks + 1)
        
        for i in range(num_docks):
            x = warehouse_dimensions.x + dock_spacing * (i + 1) - dock_width / 2
            y = warehouse_dimensions.y
            
            dock = Dock(
                id=f"dock_{i+1}",
                position=Point(x, y),
                width=dock_width,
                depth=dock_depth,
                dock_type="receiving" if i % 2 == 0 else "shipping"
            )
            
            layout['docks'].append(dock)
    
    def _place_aisles(self, 
                     layout: Dict[str, List[WarehouseElement]], 
                     warehouse_dimensions: Rectangle, 
                     num_aisles: int):
        """Place aisles in the warehouse."""
        aisle_width = self.config.get('aisle_width', 3.0)
        
        # Calculate the spacing between aisles
        usable_width = warehouse_dimensions.width
        aisle_spacing = usable_width / (num_aisles + 1)
        
        # Create horizontal aisles
        for i in range(num_aisles):
            x = warehouse_dimensions.x
            y = warehouse_dimensions.y + aisle_spacing * (i + 1)
            
            aisle = Aisle(
                id=f"aisle_h_{i+1}",
                position=Point(x, y),
                width=warehouse_dimensions.width,
                depth=aisle_width,
                orientation="horizontal"
            )
            
            layout['aisles'].append(aisle)
            
        # Create a central vertical aisle
        central_aisle = Aisle(
            id="aisle_v_central",
            position=Point(warehouse_dimensions.width / 2 - aisle_width / 2, warehouse_dimensions.y),
            width=aisle_width,
            depth=warehouse_dimensions.height,
            orientation="vertical"
        )
        
        layout['aisles'].append(central_aisle)
    
    def _place_racks(self, 
                    layout: Dict[str, List[WarehouseElement]], 
                    warehouse_dimensions: Rectangle, 
                    num_racks: int):
        """Place storage racks between aisles."""
        rack_width = self.config.get('rack_width', 2.5)
        rack_depth = self.config.get('rack_depth', 1.2)
        
        horizontal_aisles = [a for a in layout['aisles'] if a.orientation == "horizontal"]
        
        # Place racks between horizontal aisles
        for i in range(len(horizontal_aisles) - 1):
            aisle1 = horizontal_aisles[i]
            aisle2 = horizontal_aisles[i + 1]
            
            # Calculate space between aisles
            space_height = aisle2.position.y - (aisle1.position.y + aisle1.depth)
            
            # Determine how many racks can fit in this space
            racks_per_row = int(warehouse_dimensions.width / (rack_width * 1.2))
            
            # Place racks in a grid between these aisles
            for j in range(racks_per_row):
                x = warehouse_dimensions.x + j * rack_width * 1.2
                y = aisle1.position.y + aisle1.depth + space_height / 2 - rack_depth / 2
                
                rack = Rack(
                    id=f"rack_{i}_{j}",
                    position=Point(x, y),
                    width=rack_width,
                    depth=rack_depth,
                    height=self.config.get('rack_height', 5.0),
                    num_shelves=self.config.get('shelves_per_rack', 3)
                )
                
                layout['racks'].append(rack)
                
                # Add shelves to this rack
                for shelf_idx in range(rack.num_shelves):
                    shelf = Shelf(
                        id=f"shelf_{i}_{j}_{shelf_idx}",
                        position=Point(x, y),
                        width=rack_width,
                        depth=rack_depth,
                        height=0.4,
                        level=shelf_idx,
                        parent_rack_id=rack.id
                    )
                    layout['shelves'].append(shelf)
    
    def _place_work_areas(self, 
                         layout: Dict[str, List[WarehouseElement]], 
                         warehouse_dimensions: Rectangle, 
                         work_area_specs: List[Dict[str, Any]]):
        """Place work areas in the warehouse."""
        # Use the top section of the warehouse for work areas
        work_area_y = warehouse_dimensions.y + warehouse_dimensions.height * 0.75
        work_area_height = warehouse_dimensions.height * 0.2
        
        # Divide the width among the work areas
        num_work_areas = len(work_area_specs)
        work_area_width = warehouse_dimensions.width / num_work_areas
        
        for i, work_area_spec in enumerate(work_area_specs):
            x = warehouse_dimensions.x + i * work_area_width
            
            work_area = WorkArea(
                id=f"work_area_{i+1}",
                position=Point(x, work_area_y),
                width=work_area_width * 0.9,  # Leave some space between work areas
                depth=work_area_height,
                work_type=work_area_spec.get('type', 'assembly'),
                capacity=work_area_spec.get('capacity', 5)
            )
            
            layout['work_areas'].append(work_area)
    
    def _place_picking_stations(self, 
                              layout: Dict[str, List[WarehouseElement]], 
                              warehouse_dimensions: Rectangle, 
                              picking_station_specs: List[Dict[str, Any]]):
        """Place picking stations near aisles for efficient order picking."""
        station_width = self.config.get('picking_station_width', 2.0)
        station_depth = self.config.get('picking_station_depth', 2.0)
        
        # Place picking stations along the vertical aisle
        vertical_aisles = [a for a in layout['aisles'] if a.orientation == "vertical"]
        
        if vertical_aisles:
            vertical_aisle = vertical_aisles[0]
            
            # Place picking stations along the aisle
            num_stations = len(picking_station_specs)
            spacing = warehouse_dimensions.height / (num_stations + 1)
            
            for i, station_spec in enumerate(picking_station_specs):
                x = vertical_aisle.position.x + vertical_aisle.width + 1.0  # Place to the right of the aisle
                y = warehouse_dimensions.y + spacing * (i + 1) - station_depth / 2
                
                station = PickingStation(
                    id=f"picking_station_{i+1}",
                    position=Point(x, y),
                    width=station_width,
                    depth=station_depth,
                    station_type=station_spec.get('type', 'standard')
                )
                
                layout['picking_stations'].append(station)
    
    def _fix_constraint_violations(self, 
                                 layout: Dict[str, List[WarehouseElement]], 
                                 violations: List[str]) -> Dict[str, List[WarehouseElement]]:
        """
        Attempt to fix constraint violations in the layout.
        
        Args:
            layout: The layout with violations
            violations: List of violation descriptions
            
        Returns:
            The fixed layout (or the original if can't be fixed)
        """
        # This is a simple implementation that tries to fix common issues
        # A more sophisticated version would analyze each violation and apply specific fixes
        
        # Apply some spacing to reduce overlaps
        for element_type in layout:
            elements = layout[element_type]
            for i, elem1 in enumerate(elements):
                for j, elem2 in enumerate(elements[i+1:], i+1):
                    if self._check_overlap(elem1, elem2):
                        # Move elem2 slightly to reduce overlap
                        shift_x = random.uniform(0.5, 1.5)
                        shift_y = random.uniform(0.5, 1.5)
                        new_pos = Point(elem2.position.x + shift_x, elem2.position.y + shift_y)
                        elem2.position = new_pos
        
        return layout
    
    def _check_overlap(self, elem1: WarehouseElement, elem2: WarehouseElement) -> bool:
        """Check if two elements overlap."""
        # Bounding box overlap check
        x1_min, y1_min = elem1.position.x, elem1.position.y
        x1_max, y1_max = x1_min + elem1.width, y1_min + elem1.depth
        
        x2_min, y2_min = elem2.position.x, elem2.position.y
        x2_max, y2_max = x2_min + elem2.width, y2_min + elem2.depth
        
        return not (x1_max < x2_min or x2_max < x1_min or
                   y1_max < y2_min or y2_max < y1_min)


class GeneticLayoutGenerator(LayoutGenerator):
    """Generate warehouse layouts using genetic algorithms."""
    
    def __init__(self, 
                 layout_engine: LayoutEngine, 
                 constraint_manager: ConstraintManager,
                 config: Dict[str, Any]):
        """Initialize the genetic layout generator."""
        super().__init__(layout_engine, constraint_manager, config)
        self.population_size = config.get('population_size', 50)
        self.num_generations = config.get('num_generations', 100)
        self.mutation_rate = config.get('mutation_rate', 0.1)
        self.crossover_rate = config.get('crossover_rate', 0.8)
        
    def generate_layout(self, warehouse_dimensions: Rectangle, 
                       requirements: Dict[str, Any]) -> Dict[str, List[WarehouseElement]]:
        """
        Generate a warehouse layout using genetic algorithms.
        
        Args:
            warehouse_dimensions: The dimensions of the warehouse
            requirements: The requirements for the warehouse layout
            
        Returns:
            A dictionary containing lists of warehouse elements by type
        """
        logger.info("Generating layout using genetic algorithm")
        
        # Create initial population
        population = self._create_initial_population(warehouse_dimensions, requirements)
        
        # Evaluate initial population
        fitness_scores = [self.evaluate_layout(layout) for layout in population]
        
        # Run genetic algorithm for specified number of generations
        best_layout = None
        best_fitness = float('-inf')
        
        for generation in range(self.num_generations):
            # Selection
            parents = self._selection(population, fitness_scores)
            
            # Create new population
            new_population = []
            
            while len(new_population) < self.population_size:
                # Select parents
                parent1, parent2 = random.sample(parents, 2)
                
                # Crossover
                if random.random() < self.crossover_rate:
                    child = self._crossover(parent1, parent2)
                else:
                    child = random.choice([parent1, parent2]).copy()
                
                # Mutation
                if random.random() < self.mutation_rate:
                    child = self._mutate(child, warehouse_dimensions)
                
                new_population.append(child)
            
            # Update population
            population = new_population
            
            # Evaluate new population
            fitness_scores = [self.evaluate_layout(layout) for layout in population]
            
            # Track best layout
            max_fitness_idx = fitness_scores.index(max(fitness_scores))
            if fitness_scores[max_fitness_idx] > best_fitness:
                best_fitness = fitness_scores[max_fitness_idx]
                best_layout = population[max_fitness_idx]
            
            logger.info(f"Generation {generation + 1}: Best fitness = {best_fitness:.2f}")
        
        return best_layout
    
    def _create_initial_population(self, 
                                 warehouse_dimensions: Rectangle, 
                                 requirements: Dict[str, Any]) -> List[Dict[str, List[WarehouseElement]]]:
        """Create an initial population of layouts."""
        population = []
        
        # Use rule-based generator to create one good starting layout
        rule_based = RuleBasedLayoutGenerator(self.layout_engine, self.constraint_manager, self.config)
        seed_layout = rule_based.generate_layout(warehouse_dimensions, requirements)
        population.append(seed_layout)
        
        # Generate the rest randomly with variations
        for _ in range(self.population_size - 1):
            # Start with a copy of the seed layout
            layout = self._copy_layout(seed_layout)
            
            # Apply random variations
            layout = self._mutate(layout, warehouse_dimensions, mutation_strength=0.5)
            
            population.append(layout)
        
        return population
    
    def _copy_layout(self, layout: Dict[str, List[WarehouseElement]]) -> Dict[str, List[WarehouseElement]]:
        """Create a deep copy of a layout."""
        new_layout = {}
        for element_type, elements in layout.items():
            new_layout[element_type] = [elem.copy() for elem in elements]
        return new_layout
    
    def _selection(self, 
                  population: List[Dict[str, List[WarehouseElement]]], 
                  fitness_scores: List[float]) -> List[Dict[str, List[WarehouseElement]]]:
        """Select parents using tournament selection."""
        parents = []
        tournament_size = max(2, self.population_size // 10)
        
        for _ in range(self.population_size):
            # Select random candidates for tournament
            candidates = random.sample(range(self.population_size), tournament_size)
            
            # Find the candidate with the highest fitness
            best_candidate = candidates[0]
            for candidate in candidates[1:]:
                if fitness_scores[candidate] > fitness_scores[best_candidate]:
                    best_candidate = candidate
            
            parents.append(population[best_candidate])
        
        return parents
    
    def _crossover(self, 
                  parent1: Dict[str, List[WarehouseElement]], 
                  parent2: Dict[str, List[WarehouseElement]]) -> Dict[str, List[WarehouseElement]]:
        """Perform crossover between two parent layouts."""
        child = {}
        
        for element_type in parent1.keys():
            child[element_type] = []
            
            # Choose elements from either parent based on a crossover point
            if random.random() < 0.5:
                # Take first half from parent1, second half from parent2
                crossover_point = len(parent1[element_type]) // 2
                child[element_type].extend([elem.copy() for elem in parent1[element_type][:crossover_point]])
                child[element_type].extend([elem.copy() for elem in parent2[element_type][crossover_point:]])
            else:
                # Take random elements from each parent
                for elem in parent1[element_type]:
                    if random.random() < 0.5:
                        child[element_type].append(elem.copy())
                for elem in parent2[element_type]:
                    if random.random() < 0.5:
                        child[element_type].append(elem.copy())
        
        return child
    
    def _mutate(self, 
                layout: Dict[str, List[WarehouseElement]], 
                warehouse_dimensions: Rectangle,
                mutation_strength: float = 0.2) -> Dict[str, List[WarehouseElement]]:
        """Mutate a layout by making random changes."""
        new_layout = self._copy_layout(layout)
        
        for element_type, elements in new_layout.items():
            for elem in elements:
                # Randomly adjust position with probability based on mutation strength
                if random.random() < mutation_strength:
                    # Add random offset to position
                    max_offset = min(warehouse_dimensions.width, warehouse_dimensions.height) * 0.1
                    offset_x = random.uniform(-max_offset, max_offset)
                    offset_y = random.uniform(-max_offset, max_offset)
                    
                    # Make sure the element stays within bounds
                    new_x = max(warehouse_dimensions.x, 
                               min(elem.position.x + offset_x, 
                                  warehouse_dimensions.x + warehouse_dimensions.width - elem.width))
                    new_y = max(warehouse_dimensions.y, 
                               min(elem.position.y + offset_y, 
                                  warehouse_dimensions.y + warehouse_dimensions.height - elem.depth))
                    
                    elem.position = Point(new_x, new_y)
                
                # Randomly adjust dimensions with lower probability
                if random.random() < mutation_strength * 0.5:
                    # Scale dimensions slightly
                    scale_factor = random.uniform(0.9, 1.1)
                    elem.width *= scale_factor
                    elem.depth *= scale_factor
        
        return new_layout


class ReinforcementLearningLayoutGenerator(LayoutGenerator):
    """Generate warehouse layouts using reinforcement learning."""
    
    def __init__(self, 
                 layout_engine: LayoutEngine, 
                 constraint_manager: ConstraintManager,
                 config: Dict[str, Any]):
        """Initialize the reinforcement learning layout generator."""
        super().__init__(layout_engine, constraint_manager, config)
        self.num_episodes = config.get('num_episodes', 1000)
        self.epsilon = config.get('epsilon', 0.1)  # Exploration rate
        self.alpha = config.get('alpha', 0.1)      # Learning rate
        self.gamma = config.get('gamma', 0.9)      # Discount factor
        self.q_values = {}  # Simple Q-table (will be populated during training)
        
    def generate_layout(self, warehouse_dimensions: Rectangle, 
                       requirements: Dict[str, Any]) -> Dict[str, List[WarehouseElement]]:
        """
        Generate a warehouse layout using reinforcement learning.
        
        Args:
            warehouse_dimensions: The dimensions of the warehouse
            requirements: The requirements for the warehouse layout
            
        Returns:
            A dictionary containing lists of warehouse elements by type
        """
        logger.info("Generating layout using reinforcement learning")
        
        # This is a simplified implementation - a real RL solution would be more complex
        # For now, we'll use the rule-based generator to create a base layout
        rule_based = RuleBasedLayoutGenerator(self.layout_engine, self.constraint_manager, self.config)
        base_layout = rule_based.generate_layout(warehouse_dimensions, requirements)
        
        # Then use our RL approach to refine it
        optimized_layout = self._optimize_with_rl(base_layout, warehouse_dimensions)
        
        return optimized_layout
    
    def _optimize_with_rl(self, 
                         layout: Dict[str, List[WarehouseElement]], 
                         warehouse_dimensions: Rectangle) -> Dict[str, List[WarehouseElement]]:
        """Optimize a layout using reinforcement learning."""
        best_layout = self._copy_layout(layout)
        best_score = self.evaluate_layout(best_layout)
        
        current_layout = self._copy_layout(best_layout)
        
        # Simple RL approach: Try modifications and keep track of improvements
        for episode in range(self.num_episodes):
            # Decide whether to explore or exploit
            if random.random() < self.epsilon:
                # Explore: make a random change
                action = self._get_random_action()
                next_layout = self._apply_action(current_layout, action, warehouse_dimensions)
            else:
                # Exploit: choose the best action based on Q-values
                action = self._get_best_action(current_layout)
                next_layout = self._apply_action(current_layout, action, warehouse_dimensions)
            
            # Calculate reward
            current_score = self.evaluate_layout(current_layout)
            next_score = self.evaluate_layout(next_layout)
            reward = next_score - current_score
            
            # Update Q-value
            current_state = self._get_state_representation(current_layout)
            next_state = self._get_state_representation(next_layout)
            
            # Initialize Q-values if needed
            if (current_state, action) not in self.q_values:
                self.q_values[(current_state, action)] = 0.0
            
            # Get maximum Q-value for next state
            next_q_max = max([self.q_values.get((next_state, a), 0.0) 
                             for a in self._get_possible_actions()])
            
            # Q-learning update formula
            self.q_values[(current_state, action)] += self.alpha * (
                reward + self.gamma * next_q_max - self.q_values[(current_state, action)]
            )
            
            # Update current layout
            current_layout = next_layout
            
            # Check if we found a better layout
            current_score = self.evaluate_layout(current_layout)
            if current_score > best_score:
                best_score = current_score
                best_layout = self._copy_layout(current_layout)
                
                logger.info(f"Episode {episode + 1}: Found better layout with score {best_score:.2f}")
        
        return best_layout
    
    def _get_random_action(self) -> str:
        """Get a random action for the RL agent."""
        return random.choice(self._get_possible_actions())
    
    def _get_possible_actions(self) -> List[str]:
        """Get the list of possible actions for the RL agent."""
        return [
            "move_rack_north", "move_rack_south", "move_rack_east", "move_rack_west",
            "move_aisle_north", "move_aisle_south", "move_aisle_east", "move_aisle_west",
            "rotate_rack", "widen_aisle", "narrow_aisle"
        ]
    
    def _get_best_action(self, 
                        layout: Dict[str, List[WarehouseElement]]) -> str:
        """Get the best action based on Q-values."""
        state = self._get_state_representation(layout)
        
        # Find action with highest Q-value
        best_action = self._get_random_action()  # Default to random if no Q-values
        best_q = float('-inf')
        
        for action in self._get_possible_actions():
            q_value = self.q_values.get((state, action), 0.0)
            if q_value > best_q:
                best_q = q_value
                best_action = action
        
        return best_action
    
    def _apply_action(self, 
                     layout: Dict[str, List[WarehouseElement]], 
                     action: str,
                     warehouse_dimensions: Rectangle) -> Dict[str, List[WarehouseElement]]:
        """Apply an action to the layout."""
        new_layout = self._copy_layout(layout)
        
        # Movement distance
        distance = 0.5  # meters
        
        if action.startswith("move_rack"):
            if not new_layout['racks']:
                return new_layout
                
            # Select a random rack
            rack = random.choice(new_layout['racks'])
            
            # Move it in the specified direction
            if action == "move_rack_north":
                rack.position.y = min(rack.position.y + distance, 
                                     warehouse_dimensions.y + warehouse_dimensions.height - rack.depth)
            elif action == "move_rack_south":
                rack.position.y = max(rack.position.y - distance, warehouse_dimensions.y)
            elif action == "move_rack_east":
                rack.position.x = min(rack.position.x + distance, 
                                     warehouse_dimensions.x + warehouse_dimensions.width - rack.width)
            elif action == "move_rack_west":
                rack.position.x = max(rack.position.x - distance, warehouse_dimensions.x)
                
        elif action.startswith("move_aisle"):
            if not new_layout['aisles']:
                return new_layout
                
            # Select a random aisle
            aisle = random.choice(new_layout['aisles'])
            
            # Move it in the specified direction
            if action == "move_aisle_north":
                aisle.position.y = min(aisle.position.y + distance, 
                                      warehouse_dimensions.y + warehouse_dimensions.height - aisle.depth)
            elif action == "move_aisle_south":
                aisle.position.y = max(aisle.position.y - distance, warehouse_dimensions.y)
            elif action == "move_aisle_east":
                aisle.position.x = min(aisle.position.x + distance, 
                                      warehouse_dimensions.x + warehouse_dimensions.width - aisle.width)
            elif action == "move_aisle_west":
                aisle.position.x = max(aisle.position.x - distance, warehouse_dimensions.x)
                
        elif action == "rotate_rack":
            if not new_layout['racks']:
                return new_layout
                
            # Select a random rack and swap its width and depth
            rack = random.choice(new_layout['racks'])
            rack.width, rack.depth = rack.depth, rack.width
            
        elif action == "widen_aisle":
            if not new_layout['aisles']:
                return new_layout
                
            # Select a random aisle and increase its width/depth
            aisle = random.choice(new_layout['aisles'])
            if aisle.orientation == "horizontal":
                aisle.depth = min(aisle.depth * 1.1, 5.0)  # Max 5m wide
            else:
                aisle.width = min(aisle.width * 1.1, 5.0)  # Max 5m wide
                
        elif action == "narrow_aisle":
            if not new_layout['aisles']:
                return new_layout
                
            # Select a random aisle and decrease its width/depth
            aisle = random.choice(new_layout['aisles'])
            if aisle.orientation == "horizontal":
                aisle.depth = max(aisle.depth * 0.9, 2.0)  # Min 2m wide
            else:
                aisle.width = max(aisle.width * 0.9, 2.0)  # Min 2m wide
        
        return new_layout
    
    def _get_state_representation(self, layout: Dict[str, List[WarehouseElement]]) -> str:
        """
        Convert a layout to a simplified state representation for the Q-table.
        
        This is a simplified version that just counts elements and their approximate positions.
        """
        # This is a simplified state representation that doesn't capture full layout complexity
        # A production system would use a more sophisticated state representation
        
        # Count elements by quadrant (divide warehouse into 4 sections)
        quadrants = {
            'racks': [0, 0, 0, 0],
            'aisles': [0, 0, 0, 0],
            'docks': [0, 0, 0, 0],
            'work_areas': [0, 0, 0, 0]
        }
        
        # Use an arbitrary warehouse size for this example
        warehouse_center_x = 50.0  # Arbitrary center point
        warehouse_center_y = 50.0  # Arbitrary center point
        
        for element_type in ['racks', 'aisles', 'docks', 'work_areas']:
            if element_type not in layout:
                continue
                
            for elem in layout[element_type]:
                # Determine quadrant
                quadrant_idx = 0
                if elem.position.x >= warehouse_center_x:
                    if elem.position.y >= warehouse_center_y:
                        quadrant_idx = 0  # top-right
                    else:
                        quadrant_idx = 3  # bottom-right
                else:
                    if elem.position.y >= warehouse_center_y:
                        quadrant_idx = 1  # top-left
                    else:
                        quadrant_idx = 2  # bottom-left
                
                quadrants[element_type][quadrant_idx] += 1
        
        # Convert to a string representation
        state_parts = []
        for element_type, counts in quadrants.items():
            state_parts.append(f"{element_type}:{','.join(map(str, counts))}")
        
        return "|".join(state_parts)
    
    def _copy_layout(self, layout: Dict[str, List[WarehouseElement]]) -> Dict[str, List[WarehouseElement]]:
        """Create a deep copy of a layout."""
        new_layout = {}
        for element_type, elements in layout.items():
            new_layout[element_type] = [elem.copy() for elem in elements]
        return new_layout


class LayoutGeneratorFactory:
    """Factory class to create appropriate layout generators."""
    
    @staticmethod
    def create_generator(generator_type: str,
                       layout_engine: LayoutEngine,
                       constraint_manager: ConstraintManager,
                       config: Dict[str, Any]) -> LayoutGenerator:
        """
        Create a layout generator of the specified type.
        
        Args:
            generator_type: The type of generator to create ('rule_based', 'genetic', or 'rl')
            layout_engine: The layout engine to use
            constraint_manager: The constraint manager to use
            config: Configuration parameters
            
        Returns:
            A layout generator instance
            
        Raises:
            ValueError: If an invalid generator type is specified
        """
        if generator_type == 'rule_based':
            return RuleBasedLayoutGenerator(layout_engine, constraint_manager, config)
        elif generator_type == 'genetic':
            return GeneticLayoutGenerator(layout_engine, constraint_manager, config)
        elif generator_type == 'rl':
            return ReinforcementLearningLayoutGenerator(layout_engine, constraint_manager, config)
        else:
            raise ValueError(f"Unknown layout generator type: {generator_type}")


class HybridLayoutGenerator(LayoutGenerator):
    """
    Generate warehouse layouts using a hybrid approach combining multiple techniques.
    This generator uses rule-based methods to create an initial layout, followed by
    optimization using either genetic algorithms or reinforcement learning.
    """
    
    def __init__(self, 
                 layout_engine: LayoutEngine, 
                 constraint_manager: ConstraintManager,
                 config: Dict[str, Any]):
        """
        Initialize the hybrid layout generator.
        
        Args:
            layout_engine: The layout engine instance to use
            constraint_manager: The constraint manager to validate layouts
            config: Configuration parameters
        """
        super().__init__(layout_engine, constraint_manager, config)
        self.optimization_method = config.get('optimization_method', 'genetic')
        
    def generate_layout(self, warehouse_dimensions: Rectangle, 
                       requirements: Dict[str, Any]) -> Dict[str, List[WarehouseElement]]:
        """
        Generate a warehouse layout using a hybrid approach.
        
        Args:
            warehouse_dimensions: The dimensions of the warehouse
            requirements: The requirements for the warehouse layout
            
        Returns:
            A dictionary containing lists of warehouse elements by type
        """
        logger.info(f"Generating hybrid layout using {self.optimization_method} optimization")
        
        # Step 1: Create initial layout using rule-based approach
        rule_generator = RuleBasedLayoutGenerator(
            self.layout_engine, self.constraint_manager, self.config
        )
        initial_layout = rule_generator.generate_layout(warehouse_dimensions, requirements)
        
        # Step 2: Optimize using the specified method
        if self.optimization_method == 'genetic':
            genetic_config = self.config.copy()
            genetic_config['population_size'] = genetic_config.get('population_size', 20)
            genetic_config['num_generations'] = genetic_config.get('num_generations', 50)
            
            optimizer = GeneticLayoutGenerator(
                self.layout_engine, self.constraint_manager, genetic_config
            )
            
            # Create a population based on the initial layout
            population = [initial_layout]
            for _ in range(genetic_config['population_size'] - 1):
                # Create variations of the initial layout
                variation = optimizer._copy_layout(initial_layout)
                variation = optimizer._mutate(variation, warehouse_dimensions, mutation_strength=0.3)
                population.append(variation)
            
            # Run a few generations of optimization
            fitness_scores = [optimizer.evaluate_layout(layout) for layout in population]
            
            best_layout = initial_layout
            best_fitness = optimizer.evaluate_layout(initial_layout)
            
            for generation in range(genetic_config['num_generations']):
                # Selection
                parents = optimizer._selection(population, fitness_scores)
                
                # Create new population
                new_population = []
                
                while len(new_population) < genetic_config['population_size']:
                    # Select parents
                    parent1, parent2 = random.sample(parents, 2)
                    
                    # Crossover
                    if random.random() < optimizer.crossover_rate:
                        child = optimizer._crossover(parent1, parent2)
                    else:
                        child = random.choice([parent1, parent2])
                    
                    # Mutation
                    if random.random() < optimizer.mutation_rate:
                        child = optimizer._mutate(child, warehouse_dimensions)
                    
                    new_population.append(child)
                
                # Update population
                population = new_population
                
                # Evaluate new population
                fitness_scores = [optimizer.evaluate_layout(layout) for layout in population]
                
                # Track best layout
                max_fitness_idx = fitness_scores.index(max(fitness_scores))
                if fitness_scores[max_fitness_idx] > best_fitness:
                    best_fitness = fitness_scores[max_fitness_idx]
                    best_layout = population[max_fitness_idx]
                
                logger.info(f"Generation {generation + 1}: Best fitness = {best_fitness:.2f}")
            
            optimized_layout = best_layout
            
        elif self.optimization_method == 'rl':
            rl_config = self.config.copy()
            rl_config['num_episodes'] = rl_config.get('num_episodes', 500)
            
            optimizer = ReinforcementLearningLayoutGenerator(
                self.layout_engine, self.constraint_manager, rl_config
            )
            
            # Use the RL optimizer to refine the layout
            optimized_layout = optimizer._optimize_with_rl(initial_layout, warehouse_dimensions)
            
        else:
            # Default to using the space optimizer for simple refinement
            optimized_layout = self.space_optimizer.optimize(initial_layout, iterations=100)
        
        # Step 3: Final validation and adjustment
        valid, violations = self.constraint_manager.validate_layout(optimized_layout)
        if not valid:
            logger.warning(f"Final layout has {len(violations)} constraint violations")
            # Try to fix remaining constraint violations
            if self.optimization_method == 'genetic':
                final_layout = optimizer._fix_constraint_violations(optimized_layout, violations)
            else:
                rule_generator = RuleBasedLayoutGenerator(
                    self.layout_engine, self.constraint_manager, self.config
                )
                final_layout = rule_generator._fix_constraint_violations(optimized_layout, violations)
        else:
            final_layout = optimized_layout
        
        return final_layout


# Helper function to use as public API for this module
def generate_warehouse_layout(
    layout_engine: LayoutEngine,
    constraint_manager: ConstraintManager,
    warehouse_dimensions: Rectangle,
    requirements: Dict[str, Any],
    generator_type: str = 'rule_based',
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, List[WarehouseElement]]:
    """
    Generate a warehouse layout using the specified generator type.
    
    Args:
        layout_engine: The layout engine to use
        constraint_manager: The constraint manager to use
        warehouse_dimensions: The dimensions of the warehouse
        requirements: The requirements for the warehouse layout
        generator_type: The type of generator to use ('rule_based', 'genetic', 'rl', or 'hybrid')
        config: Configuration parameters for the generator
        
    Returns:
        A dictionary containing lists of warehouse elements by type
    """
    if config is None:
        config = {}
    
    generator = LayoutGeneratorFactory.create_generator(
        generator_type, layout_engine, constraint_manager, config
    )
    
    return generator.generate_layout(warehouse_dimensions, requirements)