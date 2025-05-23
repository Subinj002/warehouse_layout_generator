"""
Optimization module for warehouse layout generation.

This module provides algorithms and methods for optimizing warehouse layouts
based on various constraints and objectives such as space utilization,
travel distance, workflow efficiency, and accessibility.
"""

import math
import random
from typing import List, Dict, Tuple, Optional, Any, Callable
from dataclasses import dataclass
"""
Defines the optimization problem structure for warehouse layout generation.
"""



# Type aliases for clarity
LayoutSolution = Dict[str, Any]
ObjectiveFunction = Callable[[LayoutSolution], float]
ConstraintFunction = Callable[[LayoutSolution], bool]

@dataclass
class OptimizationProblem:
    """Represents a warehouse layout optimization problem."""
    warehouse_dimensions: tuple[float, float]
    elements: List[Dict[str, Any]]
    constraints: List[Dict[str, Any]]
    objectives: List[Dict[str, Any]]

@dataclass
class OptimizationResult:
    """Represents the result of an optimization run."""
    success: bool
    layout: Dict[str, Any]
    score: float
    violations: List[str]
    iterations: int
class LayoutOptimizer:
    """Base class for warehouse layout optimization algorithms."""
    
    def __init__(self, objective_functions: Dict[str, Tuple[ObjectiveFunction, float]],
                 constraint_functions: List[ConstraintFunction]):
        """
        Initialize the layout optimizer.
        
        Args:
            objective_functions: Dictionary mapping objective names to tuples of 
                                (objective_function, weight)
            constraint_functions: List of constraint functions that must be satisfied
        """
        self.objective_functions = objective_functions
        self.constraint_functions = constraint_functions
        
    def evaluate_solution(self, solution: LayoutSolution) -> float:
        """
        Evaluate a solution based on objective functions and their weights.
        
        Args:
            solution: A layout solution to evaluate
            
        Returns:
            The weighted sum of all objective functions
        """
        # Check if all constraints are satisfied
        if not self.is_feasible(solution):
            return float('inf')  # Return infinity for infeasible solutions
        
        # Calculate weighted sum of objective functions
        total_score = 0.0
        for obj_func, weight in self.objective_functions.values():
            total_score += obj_func(solution) * weight
            
        return total_score
    
    def is_feasible(self, solution: LayoutSolution) -> bool:
        """
        Check if a solution satisfies all constraints.
        
        Args:
            solution: A layout solution to check
            
        Returns:
            True if all constraints are satisfied, False otherwise
        """
        return all(constraint(solution) for constraint in self.constraint_functions)
    
    def optimize(self, initial_solution: LayoutSolution, max_iterations: int) -> LayoutSolution:
        """
        Placeholder for the optimization algorithm.
        
        Args:
            initial_solution: Starting layout solution
            max_iterations: Maximum number of iterations
            
        Returns:
            Optimized layout solution
        """
        raise NotImplementedError("Subclasses must implement optimize method")


class SimulatedAnnealing(LayoutOptimizer):
    """Simulated Annealing implementation for warehouse layout optimization."""
    
    def optimize(self, initial_solution: LayoutSolution, max_iterations: int = 1000,
                 initial_temp: float = 100.0, cooling_rate: float = 0.95) -> LayoutSolution:
        """
        Optimize layout using simulated annealing algorithm.
        
        Args:
            initial_solution: Starting layout solution
            max_iterations: Maximum number of iterations
            initial_temp: Starting temperature for annealing process
            cooling_rate: Rate at which temperature decreases
            
        Returns:
            Optimized layout solution
        """
        current_solution = initial_solution.copy()
        best_solution = current_solution.copy()
        
        current_score = self.evaluate_solution(current_solution)
        best_score = current_score
        
        temp = initial_temp
        
        for iteration in range(max_iterations):
            # Generate neighbor solution
            neighbor = self._generate_neighbor(current_solution)
            neighbor_score = self.evaluate_solution(neighbor)
            
            # Calculate acceptance probability
            delta = neighbor_score - current_score
            acceptance_prob = math.exp(-delta / temp) if delta > 0 else 1.0
            
            # Accept or reject the neighbor solution
            if random.random() < acceptance_prob:
                current_solution = neighbor
                current_score = neighbor_score
                
                # Update best solution if needed
                if current_score < best_score:
                    best_solution = current_solution.copy()
                    best_score = current_score
            
            # Cool down the temperature
            temp *= cooling_rate
            
            # Optional: early termination if temp is too low
            if temp < 0.01:
                break
                
        return best_solution
    
    def _generate_neighbor(self, solution: LayoutSolution) -> LayoutSolution:
        """
        Generate a neighboring solution by making small modifications.
        Subclasses should override this with domain-specific neighbor generation.
        
        Args:
            solution: Current layout solution
            
        Returns:
            A neighboring layout solution
        """
        # This is a placeholder implementation - actual implementation will depend
        # on the specific representation of warehouse layouts
        neighbor = solution.copy()
        
        # Example: Swap two random elements if there are elements to swap
        if 'elements' in neighbor and len(neighbor['elements']) > 1:
            idx1, idx2 = random.sample(range(len(neighbor['elements'])), 2)
            neighbor['elements'][idx1], neighbor['elements'][idx2] = (
                neighbor['elements'][idx2], neighbor['elements'][idx1]
            )
            
        return neighbor


class GeneticAlgorithm(LayoutOptimizer):
    """Genetic Algorithm implementation for warehouse layout optimization."""
    
    def optimize(self, initial_population: List[LayoutSolution], max_generations: int = 100,
                 population_size: int = 50, crossover_rate: float = 0.8, 
                 mutation_rate: float = 0.2) -> LayoutSolution:
        """
        Optimize layout using genetic algorithm.
        
        Args:
            initial_population: List of starting layout solutions
            max_generations: Maximum number of generations
            population_size: Size of the population
            crossover_rate: Probability of crossover
            mutation_rate: Probability of mutation
            
        Returns:
            Optimized layout solution
        """
        # Ensure population is the right size
        population = initial_population[:population_size]
        while len(population) < population_size:
            # Clone and mutate existing solutions if we don't have enough
            idx = random.randint(0, len(initial_population) - 1)
            population.append(self._mutate(initial_population[idx].copy()))
        
        for generation in range(max_generations):
            # Evaluate all solutions
            fitness_scores = [self.evaluate_solution(solution) for solution in population]
            
            # Find the best solution in this generation
            best_idx = fitness_scores.index(min(fitness_scores))
            best_solution = population[best_idx].copy()
            
            # Create new population through selection, crossover, and mutation
            new_population = [best_solution]  # Elitism: keep the best solution
            
            while len(new_population) < population_size:
                # Selection
                parent1 = self._selection(population, fitness_scores)
                parent2 = self._selection(population, fitness_scores)
                
                # Crossover
                if random.random() < crossover_rate:
                    child = self._crossover(parent1, parent2)
                else:
                    child = parent1.copy()
                
                # Mutation
                if random.random() < mutation_rate:
                    child = self._mutate(child)
                
                # Add to new population if feasible
                if self.is_feasible(child):
                    new_population.append(child)
            
            # Replace old population
            population = new_population
        
        # Return best solution from final population
        fitness_scores = [self.evaluate_solution(solution) for solution in population]
        best_idx = fitness_scores.index(min(fitness_scores))
        return population[best_idx]
    
    def _selection(self, population: List[LayoutSolution], 
                  fitness_scores: List[float]) -> LayoutSolution:
        """
        Select a solution from the population using tournament selection.
        
        Args:
            population: List of solutions
            fitness_scores: Corresponding fitness scores
            
        Returns:
            Selected solution
        """
        # Tournament selection
        tournament_size = 3
        selected_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitness = [fitness_scores[idx] for idx in selected_indices]
        
        # Select the best from the tournament (lowest score is best)
        winner_idx = selected_indices[tournament_fitness.index(min(tournament_fitness))]
        return population[winner_idx].copy()
    
    def _crossover(self, parent1: LayoutSolution, parent2: LayoutSolution) -> LayoutSolution:
        """
        Create a new solution by combining two parent solutions.
        Subclasses should override this with domain-specific crossover.
        
        Args:
            parent1: First parent solution
            parent2: Second parent solution
            
        Returns:
            New child solution
        """
        # This is a placeholder implementation - actual implementation will depend
        # on the specific representation of warehouse layouts
        child = parent1.copy()
        
        # Example: For lists of elements, take some elements from each parent
        if 'elements' in parent1 and 'elements' in parent2:
            elements1 = parent1['elements']
            elements2 = parent2['elements']
            
            # Ensure both parents have the same number of elements
            if len(elements1) == len(elements2):
                crossover_point = random.randint(1, len(elements1) - 1)
                child['elements'] = elements1[:crossover_point] + elements2[crossover_point:]
        
        return child
    
    def _mutate(self, solution: LayoutSolution) -> LayoutSolution:
        """
        Mutate a solution by making random changes.
        Subclasses should override this with domain-specific mutation.
        
        Args:
            solution: Solution to mutate
            
        Returns:
            Mutated solution
        """
        # This is a placeholder implementation - actual implementation will depend
        # on the specific representation of warehouse layouts
        mutated = solution.copy()
        
        # Example: Randomly modify positions of elements
        if 'elements' in mutated and mutated['elements']:
            idx = random.randint(0, len(mutated['elements']) - 1)
            if 'position' in mutated['elements'][idx]:
                # Slightly adjust position
                pos = mutated['elements'][idx]['position']
                mutated['elements'][idx]['position'] = (
                    pos[0] + random.uniform(-0.5, 0.5),
                    pos[1] + random.uniform(-0.5, 0.5)
                )
                
        return mutated


class ParticleSwarmOptimization(LayoutOptimizer):
    """Particle Swarm Optimization implementation for warehouse layout optimization."""
    
    def optimize(self, initial_solution: LayoutSolution, num_particles: int = 30,
                 max_iterations: int = 100, w: float = 0.7, c1: float = 1.4, 
                 c2: float = 1.4) -> LayoutSolution:
        """
        Optimize layout using particle swarm optimization.
        
        Args:
            initial_solution: Starting layout solution
            num_particles: Number of particles in the swarm
            max_iterations: Maximum number of iterations
            w: Inertia weight
            c1: Cognitive parameter
            c2: Social parameter
            
        Returns:
            Optimized layout solution
        """
        # Initialize particles
        particles = []
        velocities = []
        personal_best_positions = []
        personal_best_scores = []
        
        # Generate initial swarm
        for _ in range(num_particles):
            # Create a new particle by perturbing the initial solution
            particle = self._perturb_solution(initial_solution)
            particles.append(particle)
            
            # Initialize velocity as zero
            velocity = {k: 0.0 for k in self._get_optimization_params(particle)}
            velocities.append(velocity)
            
            # Initialize personal best
            score = self.evaluate_solution(particle)
            personal_best_positions.append(particle.copy())
            personal_best_scores.append(score)
        
        # Initialize global best
        global_best_idx = personal_best_scores.index(min(personal_best_scores))
        global_best_position = personal_best_positions[global_best_idx].copy()
        global_best_score = personal_best_scores[global_best_idx]
        
        # Main PSO loop
        for _ in range(max_iterations):
            for i in range(num_particles):
                particle = particles[i]
                velocity = velocities[i]
                
                # Update velocity and position for each parameter
                params = self._get_optimization_params(particle)
                for param in params:
                    # Calculate new velocity
                    inertia = w * velocity[param]
                    cognitive = c1 * random.random() * (
                        self._get_param_value(personal_best_positions[i], param) - 
                        self._get_param_value(particle, param)
                    )
                    social = c2 * random.random() * (
                        self._get_param_value(global_best_position, param) - 
                        self._get_param_value(particle, param)
                    )
                    
                    velocity[param] = inertia + cognitive + social
                    
                    # Update position
                    current_value = self._get_param_value(particle, param)
                    self._set_param_value(particle, param, current_value + velocity[param])
                
                # Ensure the solution remains feasible
                self._repair_solution(particle)
                
                # Evaluate new position
                score = self.evaluate_solution(particle)
                
                # Update personal best
                if score < personal_best_scores[i]:
                    personal_best_positions[i] = particle.copy()
                    personal_best_scores[i] = score
                    
                    # Update global best
                    if score < global_best_score:
                        global_best_position = particle.copy()
                        global_best_score = score
        
        return global_best_position
    
    def _perturb_solution(self, solution: LayoutSolution) -> LayoutSolution:
        """
        Create a new solution by perturbing an existing one.
        
        Args:
            solution: Base solution
            
        Returns:
            Perturbed solution
        """
        perturbed = solution.copy()
        
        # Example: Perturb positions of elements
        if 'elements' in perturbed:
            for element in perturbed['elements']:
                if 'position' in element:
                    # Add random perturbation
                    pos = element['position']
                    element['position'] = (
                        pos[0] + random.uniform(-1.0, 1.0),
                        pos[1] + random.uniform(-1.0, 1.0)
                    )
        
        return perturbed
    
    def _get_optimization_params(self, solution: LayoutSolution) -> List[str]:
        """
        Get a list of parameters that can be optimized in the solution.
        
        Args:
            solution: Layout solution
            
        Returns:
            List of parameter names
        """
        # This should be implemented based on the specific layout representation
        # Example: Return a list of parameter names
        params = []
        
        if 'elements' in solution:
            for i, element in enumerate(solution['elements']):
                if 'position' in element:
                    params.append(f"element_{i}_x")
                    params.append(f"element_{i}_y")
                if 'rotation' in element:
                    params.append(f"element_{i}_rotation")
        
        return params
    
    def _get_param_value(self, solution: LayoutSolution, param: str) -> float:
        """
        Extract the value of a specific parameter from the solution.
        
        Args:
            solution: Layout solution
            param: Parameter name
            
        Returns:
            Parameter value
        """
        # Parse parameter name to find the element and property
        parts = param.split('_')
        
        if parts[0] == 'element' and len(parts) >= 4:
            element_idx = int(parts[1])
            property_name = parts[2]
            
            if 'elements' in solution and element_idx < len(solution['elements']):
                element = solution['elements'][element_idx]
                
                if property_name == 'x' and 'position' in element:
                    return element['position'][0]
                elif property_name == 'y' and 'position' in element:
                    return element['position'][1]
                elif property_name == 'rotation' and 'rotation' in element:
                    return element['rotation']
        
        return 0.0  # Default value if parameter not found
    
    def _set_param_value(self, solution: LayoutSolution, param: str, value: float) -> None:
        """
        Set the value of a specific parameter in the solution.
        
        Args:
            solution: Layout solution
            param: Parameter name
            value: New parameter value
        """
        # Parse parameter name to find the element and property
        parts = param.split('_')
        
        if parts[0] == 'element' and len(parts) >= 4:
            element_idx = int(parts[1])
            property_name = parts[2]
            
            if 'elements' in solution and element_idx < len(solution['elements']):
                element = solution['elements'][element_idx]
                
                if property_name == 'x' and 'position' in element:
                    element['position'] = (value, element['position'][1])
                elif property_name == 'y' and 'position' in element:
                    element['position'] = (element['position'][0], value)
                elif property_name == 'rotation' and 'rotation' in element:
                    element['rotation'] = value
    
    def _repair_solution(self, solution: LayoutSolution) -> None:
        """
        Repair a solution to ensure it meets basic requirements.
        
        Args:
            solution: Layout solution to repair
        """
        # Example: Ensure all elements are within warehouse bounds
        if 'elements' in solution and 'warehouse_dimensions' in solution:
            width, height = solution['warehouse_dimensions']
            
            for element in solution['elements']:
                if 'position' in element:
                    x, y = element['position']
                    # Keep elements inside the warehouse with some margin
                    margin = 0.5  # Half-unit margin
                    element['position'] = (
                        max(margin, min(x, width - margin)),
                        max(margin, min(y, height - margin))
                    )


# Common objective functions for warehouse optimization

def space_utilization_objective(solution: LayoutSolution) -> float:
    """
    Calculate the unused space in the warehouse layout.
    Lower values are better.
    
    Args:
        solution: Layout solution
        
    Returns:
        Unused space score (lower is better)
    """
    if 'warehouse_dimensions' not in solution or 'elements' not in solution:
        return float('inf')
    
    total_area = solution['warehouse_dimensions'][0] * solution['warehouse_dimensions'][1]
    used_area = sum(element.get('area', 0) for element in solution['elements'])
    
    # Return percentage of unused space (lower is better)
    return (total_area - used_area) / total_area * 100


def travel_distance_objective(solution: LayoutSolution) -> float:
    """
    Calculate the total travel distance between connected elements.
    Lower values are better.
    
    Args:
        solution: Layout solution
        
    Returns:
        Total travel distance (lower is better)
    """
    if 'elements' not in solution or 'connections' not in solution:
        return float('inf')
    
    total_distance = 0.0
    
    for conn in solution['connections']:
        from_idx, to_idx, weight = conn['from'], conn['to'], conn.get('weight', 1.0)
        
        if (from_idx < len(solution['elements']) and 
            to_idx < len(solution['elements']) and
            'position' in solution['elements'][from_idx] and
            'position' in solution['elements'][to_idx]):
            
            pos1 = solution['elements'][from_idx]['position']
            pos2 = solution['elements'][to_idx]['position']
            
            # Calculate Euclidean distance
            distance = math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
            total_distance += distance * weight
    
    return total_distance


def adjacency_objective(solution: LayoutSolution) -> float:
    """
    Calculate how well the layout satisfies adjacency requirements.
    Lower values are better.
    
    Args:
        solution: Layout solution
        
    Returns:
        Adjacency score (lower is better)
    """
    if ('elements' not in solution or 
        'adjacency_requirements' not in solution):
        return float('inf')
    
    total_penalty = 0.0
    
    for req in solution['adjacency_requirements']:
        from_idx, to_idx, importance = req['from'], req['to'], req.get('importance', 1.0)
        
        if (from_idx < len(solution['elements']) and 
            to_idx < len(solution['elements']) and
            'position' in solution['elements'][from_idx] and
            'position' in solution['elements'][to_idx]):
            
            pos1 = solution['elements'][from_idx]['position']
            pos2 = solution['elements'][to_idx]['position']
            
            # Calculate Euclidean distance
            distance = math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
            
            # Penalty increases with distance and importance
            total_penalty += distance * importance
    
    return total_penalty


# Sample constraint functions

def boundary_constraint(solution: LayoutSolution) -> bool:
    """
    Check if all elements are within the warehouse boundaries.
    
    Args:
        solution: Layout solution
        
    Returns:
        True if all elements are within boundaries, False otherwise
    """
    if 'warehouse_dimensions' not in solution or 'elements' not in solution:
        return False
    
    width, height = solution['warehouse_dimensions']
    
    for element in solution['elements']:
        if 'position' not in element or 'dimensions' not in element:
            continue
            
        x, y = element['position']
        w, h = element['dimensions']
        
        # Check if the element is completely within the boundaries
        if (x < 0 or y < 0 or 
            x + w > width or 
            y + h > height):
            return False
    
    return True


def overlap_constraint(solution: LayoutSolution) -> bool:
    """
    Check if any elements overlap with each other.
    
    Args:
        solution: Layout solution
        
    Returns:
        True if no elements overlap, False otherwise
    """
    if 'elements' not in solution:
        return False
    
    elements = solution['elements']
    
    for i in range(len(elements)):
        if 'position' not in elements[i] or 'dimensions' not in elements[i]:
            continue
            
        x1, y1 = elements[i]['position']
        w1, h1 = elements[i]['dimensions']
        
        for j in range(i + 1, len(elements)):
            if 'position' not in elements[j] or 'dimensions' not in elements[j]:
                continue
                
            x2, y2 = elements[j]['position']
            w2, h2 = elements[j]['dimensions']
            
            # Check for overlap using axis-aligned bounding box test
            if (x1 < x2 + w2 and
                x1 + w1 > x2 and
                y1 < y2 + h2 and
                y1 + h1 > y2):
                return False
    
    return True


def zone_constraint(solution: LayoutSolution) -> bool:
    """
    Check if elements are placed in their required zones.
    
    Args:
        solution: Layout solution
        
    Returns:
        True if all elements are in their required zones, False otherwise
    """
    if ('elements' not in solution or 
        'zones' not in solution):
        return True  # No zones defined, constraint is satisfied
    
    for element in solution['elements']:
        if ('position' not in element or 
            'required_zone' not in element):
            continue
            
        element_x, element_y = element['position']
        required_zone_id = element['required_zone']
        
        # Find the required zone
        required_zone = None
        for zone in solution['zones']:
            if zone['id'] == required_zone_id:
                required_zone = zone
                break
                
        if required_zone is None:
            continue  # Zone not found, skip this element
            
        # Check if the element is within the zone
        zone_x, zone_y = required_zone['position']
        zone_width, zone_height = required_zone['dimensions']
        
        if not (zone_x <= element_x and 
                element_x <= zone_x + zone_width and
                zone_y <= element_y and 
                element_y <= zone_y + zone_height):
            return False
    
    return True


# Factory function to create optimizer based on configuration
def create_optimizer(config: Dict[str, Any]) -> LayoutOptimizer:
    """
    Create an optimizer instance based on configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        LayoutOptimizer instance
    """
    # Extract objective functions
    objective_functions = {}
    for obj_name, obj_config in config.get('objectives', {}).items():
        # Map objective names to functions
        obj_func = None
        if obj_name == 'space_utilization':
            obj_func = space_utilization_objective
        elif obj_name == 'travel_distance':
            obj_func = travel_distance_objective
        elif obj_name == 'adjacency':
            obj_func = adjacency_objective
        
        if obj_func:
            weight = obj_config.get('weight', 1.0)
            objective_functions[obj_name] = (obj_func, weight)
    
    # Extract constraint functions
    constraint_functions = []
    for const_name in config.get('constraints', []):
        # Map constraint names to functions
        if const_name == 'boundary':
            constraint_functions.append(boundary_constraint)
        elif const_name == 'overlap':
            constraint_functions.append(overlap_constraint)
        elif const_name == 'zone':
            constraint_functions.append(zone_constraint)
    
    # Create optimizer based on specified algorithm
    algorithm = config.get('algorithm', 'simulated_annealing')
    
    if algorithm == 'simulated_annealing':
        return SimulatedAnnealing(objective_functions, constraint_functions)
    elif algorithm == 'genetic_algorithm':
        return GeneticAlgorithm(objective_functions, constraint_functions)
    elif algorithm == 'particle_swarm':
        return ParticleSwarmOptimization(objective_functions, constraint_functions)
    else:
        raise ValueError(f"Unknown optimization algorithm: {algorithm}")
