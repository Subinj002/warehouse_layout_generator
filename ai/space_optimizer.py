"""
Space Optimizer Module

This module contains algorithms and classes for optimizing warehouse space utilization.
It uses various optimization techniques to find the most efficient arrangement of
warehouse elements within the given constraints.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import logging
from dataclasses import dataclass

from core.warehouse_elements import WarehouseElement, Rack, Aisle, WorkStation, DockDoor
from core.constraints import ConstraintManager, ConstraintViolation
from core.optimization import OptimizationProblem, OptimizationResult
from utils.geometry import Rectangle, Point, calculate_distance, check_overlap

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class SpaceAllocation:
    """Represents a space allocation for a warehouse element."""
    element: WarehouseElement
    position: Point
    rotation: float  # Rotation in degrees
    score: float


class SpaceOptimizer:
    """Main class for optimizing warehouse space allocation."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the space optimizer with configuration.
        
        Args:
            config: Dictionary containing optimization parameters
        """
        self.config = config
        self.constraint_manager = ConstraintManager()
        self.warehouse_dimensions = Rectangle(
            x=0, 
            y=0, 
            width=config.get('warehouse_width', 100),
            height=config.get('warehouse_length', 100)
        )
        self.optimization_iterations = config.get('optimization_iterations', 1000)
        self.temperature = config.get('initial_temperature', 100.0)
        self.cooling_rate = config.get('cooling_rate', 0.95)
        
    def optimize_layout(self, elements: List[WarehouseElement]) -> List[SpaceAllocation]:
        """
        Optimize the layout of warehouse elements.
        
        Args:
            elements: List of warehouse elements to position
            
        Returns:
            List of space allocations with optimized positions
        """
        logger.info(f"Starting space optimization for {len(elements)} elements")
        
        # Initial random placement
        current_solution = self._generate_initial_solution(elements)
        current_score = self._evaluate_solution(current_solution)
        
        best_solution = current_solution.copy()
        best_score = current_score
        
        # Simulated annealing optimization
        temp = self.temperature
        for iteration in range(self.optimization_iterations):
            # Generate a neighbor solution by modifying the current one
            neighbor_solution = self._generate_neighbor(current_solution)
            neighbor_score = self._evaluate_solution(neighbor_solution)
            
            # Decide whether to accept the new solution
            if self._accept_solution(current_score, neighbor_score, temp):
                current_solution = neighbor_solution
                current_score = neighbor_score
                
                # Update best solution if needed
                if current_score > best_score:
                    best_solution = current_solution.copy()
                    best_score = current_score
                    logger.info(f"New best solution found at iteration {iteration} with score {best_score:.2f}")
            
            # Cool down the temperature
            temp *= self.cooling_rate
            
            if iteration % 100 == 0:
                logger.debug(f"Iteration {iteration}, temp: {temp:.2f}, score: {current_score:.2f}, best: {best_score:.2f}")
        
        logger.info(f"Space optimization completed. Final score: {best_score:.2f}")
        return best_solution
    
    def _generate_initial_solution(self, elements: List[WarehouseElement]) -> List[SpaceAllocation]:
        """Generate initial random placement of elements."""
        solution = []
        
        # Place fixed elements first (e.g., dock doors)
        fixed_elements = [e for e in elements if getattr(e, 'fixed_position', False)]
        movable_elements = [e for e in elements if not getattr(e, 'fixed_position', False)]
        
        # Add fixed elements with their predefined positions
        for element in fixed_elements:
            if hasattr(element, 'position') and element.position:
                solution.append(SpaceAllocation(
                    element=element,
                    position=element.position,
                    rotation=getattr(element, 'rotation', 0.0),
                    score=0.0  # Will be calculated later
                ))
            else:
                logger.warning(f"Fixed element {element.id} has no position defined")
        
        # Randomly place movable elements
        for element in movable_elements:
            position = self._find_random_valid_position(element, solution)
            rotation = np.random.choice([0, 90, 180, 270]) if element.can_rotate else 0
            
            solution.append(SpaceAllocation(
                element=element,
                position=position,
                rotation=rotation,
                score=0.0  # Will be calculated later
            ))
        
        return solution
    
    def _find_random_valid_position(self, element: WarehouseElement, 
                                   existing_allocations: List[SpaceAllocation]) -> Point:
        """Find a random position that doesn't overlap with existing elements."""
        max_attempts = 100
        
        for _ in range(max_attempts):
            # Generate random position within warehouse bounds
            x = np.random.uniform(0, self.warehouse_dimensions.width - element.width)
            y = np.random.uniform(0, self.warehouse_dimensions.height - element.length)
            position = Point(x=x, y=y)
            
            # Check for overlaps
            if not self._has_overlap(element, position, 0, existing_allocations):
                return position
        
        # If no valid position found after max attempts, place it in the corner
        # and let the optimization algorithm move it later
        logger.warning(f"Could not find valid position for {element.id} after {max_attempts} attempts")
        return Point(x=0, y=0)
    
    def _has_overlap(self, element: WarehouseElement, position: Point, rotation: float,
                    allocations: List[SpaceAllocation]) -> bool:
        """Check if an element at the given position overlaps with existing allocations."""
        # Calculate element's rectangle based on position and rotation
        width, length = (element.length, element.width) if rotation in [90, 270] else (element.width, element.length)
        new_rect = Rectangle(x=position.x, y=position.y, width=width, height=length)
        
        # Check against each existing allocation
        for alloc in allocations:
            existing_width, existing_length = (
                (alloc.element.length, alloc.element.width) 
                if alloc.rotation in [90, 270] 
                else (alloc.element.width, alloc.element.length)
            )
            existing_rect = Rectangle(
                x=alloc.position.x, 
                y=alloc.position.y, 
                width=existing_width, 
                height=existing_length
            )
            
            if check_overlap(new_rect, existing_rect):
                return True
        
        return False
    
    def _generate_neighbor(self, solution: List[SpaceAllocation]) -> List[SpaceAllocation]:
        """Generate a neighboring solution by perturbing the current one."""
        neighbor = solution.copy()
        
        # Randomly select an element to modify (excluding fixed elements)
        movable_indices = [i for i, alloc in enumerate(neighbor) 
                          if not getattr(alloc.element, 'fixed_position', False)]
        
        if not movable_indices:
            return neighbor  # No movable elements
        
        idx = np.random.choice(movable_indices)
        
        # Choose a modification type: move, rotate, or swap
        mod_type = np.random.choice(['move', 'rotate', 'swap'])
        
        if mod_type == 'move':
            # Perturb position
            max_distance = min(self.warehouse_dimensions.width, self.warehouse_dimensions.height) * 0.1
            dx = np.random.uniform(-max_distance, max_distance)
            dy = np.random.uniform(-max_distance, max_distance)
            
            new_x = max(0, min(neighbor[idx].position.x + dx, 
                              self.warehouse_dimensions.width - neighbor[idx].element.width))
            new_y = max(0, min(neighbor[idx].position.y + dy, 
                              self.warehouse_dimensions.height - neighbor[idx].element.length))
            
            neighbor[idx] = SpaceAllocation(
                element=neighbor[idx].element,
                position=Point(x=new_x, y=new_y),
                rotation=neighbor[idx].rotation,
                score=0.0
            )
            
        elif mod_type == 'rotate' and neighbor[idx].element.can_rotate:
            # Rotate element
            rotations = [0, 90, 180, 270]
            current_rotation_idx = rotations.index(neighbor[idx].rotation)
            new_rotation_idx = (current_rotation_idx + np.random.choice([1, 3])) % 4
            
            neighbor[idx] = SpaceAllocation(
                element=neighbor[idx].element,
                position=neighbor[idx].position,
                rotation=rotations[new_rotation_idx],
                score=0.0
            )
            
        elif mod_type == 'swap' and len(movable_indices) > 1:
            # Swap two elements
            other_indices = [i for i in movable_indices if i != idx]
            other_idx = np.random.choice(other_indices)
            
            # Swap positions, keeping rotations
            neighbor[idx], neighbor[other_idx] = (
                SpaceAllocation(
                    element=neighbor[idx].element,
                    position=neighbor[other_idx].position,
                    rotation=neighbor[idx].rotation,
                    score=0.0
                ),
                SpaceAllocation(
                    element=neighbor[other_idx].element,
                    position=neighbor[idx].position,
                    rotation=neighbor[other_idx].rotation,
                    score=0.0
                )
            )
        
        return neighbor
    
    def _evaluate_solution(self, solution: List[SpaceAllocation]) -> float:
        """Evaluate the quality of a solution with a score."""
        # Initialize with base score
        total_score = 100.0
        
        # Check constraint violations
        violations = self._check_constraints(solution)
        constraint_penalty = sum(v.severity for v in violations)
        
        # Apply metrics
        efficiency_score = self._calculate_space_efficiency(solution)
        flow_score = self._calculate_flow_efficiency(solution)
        proximity_score = self._calculate_proximity_score(solution)
        
        # Combine scores and subtract penalties
        total_score = (
            efficiency_score * self.config.get('efficiency_weight', 0.4) +
            flow_score * self.config.get('flow_weight', 0.3) +
            proximity_score * self.config.get('proximity_weight', 0.3)
        ) - constraint_penalty
        
        # Update individual allocation scores
        for idx, alloc in enumerate(solution):
            # Calculate individual element score based on its position
            element_score = self._calculate_element_score(alloc, solution)
            solution[idx] = SpaceAllocation(
                element=alloc.element,
                position=alloc.position,
                rotation=alloc.rotation,
                score=element_score
            )
        
        return max(0, total_score)  # Ensure non-negative score
    
    def _check_constraints(self, solution: List[SpaceAllocation]) -> List[ConstraintViolation]:
        """Check for constraint violations in the solution."""
        violations = []
        
        # Check overlaps
        for i, alloc1 in enumerate(solution):
            # Calculate dimensions based on rotation
            width1, length1 = (
                (alloc1.element.length, alloc1.element.width) 
                if alloc1.rotation in [90, 270] 
                else (alloc1.element.width, alloc1.element.length)
            )
            rect1 = Rectangle(
                x=alloc1.position.x, 
                y=alloc1.position.y, 
                width=width1, 
                height=length1
            )
            
            # Check if out of bounds
            if (rect1.x < 0 or rect1.y < 0 or 
                rect1.x + rect1.width > self.warehouse_dimensions.width or 
                rect1.y + rect1.height > self.warehouse_dimensions.height):
                violations.append(ConstraintViolation(
                    constraint_type="boundary",
                    elements=[alloc1.element.id],
                    message=f"Element {alloc1.element.id} is out of warehouse bounds",
                    severity=50.0
                ))
            
            # Check for overlaps with other elements
            for j, alloc2 in enumerate(solution):
                if i >= j:  # Skip self and already checked pairs
                    continue
                
                width2, length2 = (
                    (alloc2.element.length, alloc2.element.width) 
                    if alloc2.rotation in [90, 270] 
                    else (alloc2.element.width, alloc2.element.length)
                )
                rect2 = Rectangle(
                    x=alloc2.position.x, 
                    y=alloc2.position.y, 
                    width=width2, 
                    height=length2
                )
                
                if check_overlap(rect1, rect2):
                    violations.append(ConstraintViolation(
                        constraint_type="overlap",
                        elements=[alloc1.element.id, alloc2.element.id],
                        message=f"Elements {alloc1.element.id} and {alloc2.element.id} overlap",
                        severity=100.0
                    ))
        
        # Check specific element type constraints
        self._check_type_specific_constraints(solution, violations)
        
        return violations
    
    def _check_type_specific_constraints(self, solution: List[SpaceAllocation], 
                                        violations: List[ConstraintViolation]) -> None:
        """Check constraints specific to element types."""
        # Find elements by type
        aisles = [a for a in solution if isinstance(a.element, Aisle)]
        racks = [r for r in solution if isinstance(r.element, Rack)]
        workstations = [w for w in solution if isinstance(w.element, WorkStation)]
        dockdoors = [d for d in solution if isinstance(d.element, DockDoor)]
        
        # Aisles must be accessible from both sides
        for aisle in aisles:
            has_access = False
            # Check if aisle connects to other aisles or to dockdoors
            for other_aisle in aisles + dockdoors:
                if aisle != other_aisle and self._elements_are_connected(aisle, other_aisle):
                    has_access = True
                    break
            
            if not has_access:
                violations.append(ConstraintViolation(
                    constraint_type="accessibility",
                    elements=[aisle.element.id],
                    message=f"Aisle {aisle.element.id} is not accessible",
                    severity=30.0
                ))
        
        # Racks must be adjacent to aisles
        for rack in racks:
            has_aisle_access = False
            for aisle in aisles:
                if self._elements_are_adjacent(rack, aisle):
                    has_aisle_access = True
                    break
            
            if not has_aisle_access:
                violations.append(ConstraintViolation(
                    constraint_type="accessibility",
                    elements=[rack.element.id],
                    message=f"Rack {rack.element.id} is not adjacent to any aisle",
                    severity=20.0
                ))
        
        # Workstations should be close to aisles
        for workstation in workstations:
            min_distance = float('inf')
            for aisle in aisles:
                dist = calculate_distance(workstation.position, aisle.position)
                min_distance = min(min_distance, dist)
            
            if min_distance > self.config.get('max_workstation_aisle_distance', 10.0):
                violations.append(ConstraintViolation(
                    constraint_type="proximity",
                    elements=[workstation.element.id],
                    message=f"Workstation {workstation.element.id} is too far from aisles",
                    severity=10.0
                ))
    
    def _elements_are_connected(self, alloc1: SpaceAllocation, alloc2: SpaceAllocation) -> bool:
        """Check if two elements are connected (touching at any point)."""
        # Implement connection check based on position and dimensions
        width1, length1 = (
            (alloc1.element.length, alloc1.element.width) 
            if alloc1.rotation in [90, 270] 
            else (alloc1.element.width, alloc1.element.length)
        )
        rect1 = Rectangle(
            x=alloc1.position.x, 
            y=alloc1.position.y, 
            width=width1, 
            height=length1
        )
        
        width2, length2 = (
            (alloc2.element.length, alloc2.element.width) 
            if alloc2.rotation in [90, 270] 
            else (alloc2.element.width, alloc2.element.length)
        )
        rect2 = Rectangle(
            x=alloc2.position.x, 
            y=alloc2.position.y, 
            width=width2, 
            height=length2
        )
        
        # Elements are connected if they touch at any point
        # This is true if one rectangle's edge touches the other's
        horizontal_touch = (
            (rect1.x <= rect2.x + rect2.width and rect1.x + rect1.width >= rect2.x) and
            (abs(rect1.y - (rect2.y + rect2.height)) < 0.01 or abs(rect2.y - (rect1.y + rect1.height)) < 0.01)
        )
        
        vertical_touch = (
            (rect1.y <= rect2.y + rect2.height and rect1.y + rect1.height >= rect2.y) and
            (abs(rect1.x - (rect2.x + rect2.width)) < 0.01 or abs(rect2.x - (rect1.x + rect1.width)) < 0.01)
        )
        
        return horizontal_touch or vertical_touch
    
    def _elements_are_adjacent(self, alloc1: SpaceAllocation, alloc2: SpaceAllocation) -> bool:
        """Check if two elements are adjacent (parallel and close)."""
        # Similar to connected but with different criteria
        return self._elements_are_connected(alloc1, alloc2)
    
    def _calculate_space_efficiency(self, solution: List[SpaceAllocation]) -> float:
        """Calculate space utilization efficiency."""
        # Calculate total used space
        total_warehouse_area = self.warehouse_dimensions.width * self.warehouse_dimensions.height
        used_area = 0.0
        
        for alloc in solution:
            width, length = (
                (alloc.element.length, alloc.element.width) 
                if alloc.rotation in [90, 270] 
                else (alloc.element.width, alloc.element.length)
            )
            used_area += width * length
        
        # Calculate efficiency score (0-100)
        target_utilization = self.config.get('target_space_utilization', 0.7)  # e.g. 70%
        current_utilization = used_area / total_warehouse_area
        
        # Score based on how close we are to target utilization
        if current_utilization <= target_utilization:
            # Below target is good, with diminishing returns as we approach target
            return 100 * (current_utilization / target_utilization)
        else:
            # Above target starts reducing score due to congestion
            return 100 * (1 - (current_utilization - target_utilization) / (1 - target_utilization))
    
    def _calculate_flow_efficiency(self, solution: List[SpaceAllocation]) -> float:
        """Calculate material flow efficiency."""
        # Simplified flow calculation based on distances between related elements
        flow_pairs = self._get_flow_pairs(solution)
        
        if not flow_pairs:
            return 50.0  # Neutral score if no flow pairs
        
        total_distance = 0
        max_possible_distance = (self.warehouse_dimensions.width**2 + self.warehouse_dimensions.height**2)**0.5
        
        for src, dst, importance in flow_pairs:
            src_position = next(a.position for a in solution if a.element.id == src)
            dst_position = next(a.position for a in solution if a.element.id == dst)
            
            distance = calculate_distance(src_position, dst_position)
            total_distance += distance * importance
        
        # Normalize and invert (shorter distances = higher score)
        avg_distance = total_distance / sum(importance for _, _, importance in flow_pairs)
        normalized_distance = min(1.0, avg_distance / max_possible_distance)
        
        return 100 * (1 - normalized_distance)
    
    def _get_flow_pairs(self, solution: List[SpaceAllocation]) -> List[Tuple[str, str, float]]:
        """Get pairs of elements with flow relationships and their importance."""
        # Example flow relationships:
        # - Dock doors to storage areas
        # - Storage areas to workstations
        # - Between workstations in a process flow
        
        flow_pairs = []
        
        # Find elements by type
        dock_doors = [a for a in solution if isinstance(a.element, DockDoor)]
        racks = [r for r in solution if isinstance(r.element, Rack)]
        workstations = [w for w in solution if isinstance(w.element, WorkStation)]
        
        # Dock doors to racks (receiving flow)
        for door in dock_doors:
            for rack in racks:
                # Higher importance for receiving docks to bulk storage
                importance = 1.0
                if hasattr(door.element, 'door_type') and door.element.door_type == 'receiving':
                    importance = 1.5
                flow_pairs.append((door.element.id, rack.element.id, importance))
        
        # Racks to workstations
        for rack in racks:
            for station in workstations:
                importance = 1.0
                flow_pairs.append((rack.element.id, station.element.id, importance))
        
        # Workstations to dock doors (shipping flow)
        for station in workstations:
            for door in dock_doors:
                importance = 1.0
                if hasattr(door.element, 'door_type') and door.element.door_type == 'shipping':
                    importance = 1.5
                flow_pairs.append((station.element.id, door.element.id, importance))
        
        return flow_pairs
    
    def _calculate_proximity_score(self, solution: List[SpaceAllocation]) -> float:
        """Calculate score based on proximity requirements."""
        proximity_pairs = self._get_proximity_pairs()
        
        if not proximity_pairs:
            return 50.0  # Neutral score if no proximity pairs
        
        total_score = 0
        
        for id1, id2, ideal_distance, importance in proximity_pairs:
            # Find positions of these elements
            try:
                pos1 = next(a.position for a in solution if a.element.id == id1)
                pos2 = next(a.position for a in solution if a.element.id == id2)
                
                actual_distance = calculate_distance(pos1, pos2)
                
                # Calculate proximity score
                distance_ratio = abs(actual_distance - ideal_distance) / ideal_distance
                pair_score = max(0, 100 * (1 - min(1, distance_ratio)))
                
                total_score += pair_score * importance
            except StopIteration:
                # Element not found, skip this pair
                continue
        
        # Normalize by total importance
        total_importance = sum(importance for _, _, _, importance in proximity_pairs)
        if total_importance > 0:
            return total_score / total_importance
        return 50.0
    
    def _get_proximity_pairs(self) -> List[Tuple[str, str, float, float]]:
        """
        Get pairs of elements with proximity requirements.
        
        Returns:
            List of tuples (id1, id2, ideal_distance, importance)
        """
        # This would typically come from configuration or user input
        # For now, return a simple example
        proximity_requirements = self.config.get('proximity_requirements', [])
        
        if not proximity_requirements:
            # Default proximity pairs if none defined
            return [
                # Example: WorkStation1 should be close to Rack1
                ("WorkStation1", "Rack1", 5.0, 1.0),
                # Example: DockDoor1 should be close to Rack2
                ("DockDoor1", "Rack2", 10.0, 0.8),
            ]
        
        return proximity_requirements
    
    def _calculate_element_score(self, alloc: SpaceAllocation, 
                               solution: List[SpaceAllocation]) -> float:
        """Calculate a score for an individual element's position."""
        # Base score
        score = 50.0
        
        # Adjust based on element type
        if isinstance(alloc.element, Rack):
            # Racks score better when close to aisles
            aisles = [a for a in solution if isinstance(a.element, Aisle)]
            if aisles:
                min_distance = min(calculate_distance(alloc.position, a.position) for a in aisles)
                aisle_proximity_score = max(0, 30 * (1 - min(1, min_distance / 10.0)))
                score += aisle_proximity_score
        
        elif isinstance(alloc.element, WorkStation):
            # Workstations score better when close to both racks and dock doors
            racks = [r for r in solution if isinstance(r.element, Rack)]
            doors = [d for d in solution if isinstance(d.element, DockDoor)]
            
            if racks:
                min_rack_distance = min(calculate_distance(alloc.position, r.position) for r in racks)
                rack_proximity_score = max(0, 20 * (1 - min(1, min_rack_distance / 15.0)))
                score += rack_proximity_score
            
            if doors:
                min_door_distance = min(calculate_distance(alloc.position, d.position) for d in doors)
                door_proximity_score = max(0, 10 * (1 - min(1, min_door_distance / 30.0)))
                score += door_proximity_score
        
        elif isinstance(alloc.element, DockDoor):
            # Dock doors score better when on the edge of the warehouse
            edge_proximity = min(
                alloc.position.x,
                alloc.position.y,
                self.warehouse_dimensions.width - alloc.position.x,
                self.warehouse_dimensions.height - alloc.position.y
            )
            edge_score = max(0, 30 * (1 - min(1, edge_proximity / 5.0)))
            score += edge_score
        
        return min(100, score)  # Cap at 100
    
    def _accept_solution(self, current_score: float, neighbor_score: float, 
                        temperature: float) -> bool:
        """Decide whether to accept a new solution based on simulated annealing criteria."""
        # Always accept better solutions
        if neighbor_score > current_score:
            return True
        
        # For worse solutions, accept with a probability based on temperature
        delta = neighbor_score - current_score
        acceptance_probability = np.exp(delta / temperature)
        
        return np.random.random() < acceptance_probability


class SpaceOptimizerFactory:
    """Factory class for creating space optimizers."""
    
    @staticmethod
    def create_optimizer(optimization_method: str, config: Dict[str, Any]) -> SpaceOptimizer:
        """
        Create an optimizer based on the specified method.
        
        Args:
            optimization_method: Method to use ('simulated_annealing', 'genetic', etc.)
            config: Configuration dictionary
            
        Returns:
            Configured space optimizer
        """
        if optimization_method == 'simulated_annealing':
            return SpaceOptimizer(config)
        # Could add more optimizer types in the future
        else:
            logger.warning(f"Unknown optimization method: {optimization_method}, using default")
            return SpaceOptimizer(config)