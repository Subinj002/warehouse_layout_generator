"""
constraints.py - Constraint definitions and validation for warehouse layouts

This module defines classes and functions to handle various constraints that apply to 
warehouse layouts, including:
- Minimum aisle widths
- Clearance zones around emergency exits and equipment
- Fire safety requirements
- Loading dock access requirements
- Building code compliance
- Workflow efficiency constraints
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from utils.geometry import Rectangle, Point, calculate_distance


class ConstraintSeverity(Enum):
    """Enum to define the severity of constraint violations"""
    CRITICAL = 3    # Must be fixed for a valid layout
    HIGH = 2        # Strongly recommended to fix
    MEDIUM = 1      # Should be addressed if possible
    LOW = 0         # Suggestions for optimization


class ConstraintViolation:
    """Class representing a constraint violation in the layout"""
    
    def __init__(self, message: str, severity: ConstraintSeverity, location: Optional[Union[Point, Rectangle]] = None,
                 elements_involved: Optional[List[str]] = None):
        """
        Initialize a constraint violation
        
        Args:
            message: Description of the violation
            severity: Severity level of the violation
            location: Optional geometric location of the violation
            elements_involved: Optional list of element IDs involved in the violation
        """
        self.message = message
        self.severity = severity
        self.location = location
        self.elements_involved = elements_involved or []
        
    def __str__(self) -> str:
        return f"{self.severity.name} violation: {self.message}"


class Constraint(ABC):
    """Base abstract class for all layout constraints"""
    
    def __init__(self, name: str, description: str, severity: ConstraintSeverity):
        """
        Initialize a constraint
        
        Args:
            name: Short name of the constraint
            description: Longer description of what the constraint checks
            severity: Default severity level for violations of this constraint
        """
        self.name = name
        self.description = description
        self.severity = severity
        
    @abstractmethod
    def validate(self, layout) -> List[ConstraintViolation]:
        """
        Validate the layout against this constraint
        
        Args:
            layout: The warehouse layout to validate
            
        Returns:
            List of constraint violations found
        """
        pass


class MinimumAisleWidthConstraint(Constraint):
    """Constraint to ensure aisles meet minimum width requirements"""
    
    def __init__(self, min_width: float, forklift_required: bool = False):
        """
        Initialize minimum aisle width constraint
        
        Args:
            min_width: Minimum aisle width in meters or feet
            forklift_required: Whether the aisles need to accommodate forklifts
        """
        super().__init__(
            name="Minimum Aisle Width",
            description=f"Ensures all aisles are at least {min_width} units wide" + 
                      (" to accommodate forklifts" if forklift_required else ""),
            severity=ConstraintSeverity.CRITICAL
        )
        self.min_width = min_width
        self.forklift_required = forklift_required
        
    def validate(self, layout) -> List[ConstraintViolation]:
        violations = []
        
        # Get all aisles from the layout
        aisles = layout.get_elements_by_type("aisle")
        
        for aisle in aisles:
            if aisle.width < self.min_width:
                violations.append(
                    ConstraintViolation(
                        message=f"Aisle {aisle.id} has width {aisle.width}, which is less than the minimum {self.min_width}",
                        severity=self.severity,
                        location=aisle.get_bounds(),
                        elements_involved=[aisle.id]
                    )
                )
                
        return violations


class EmergencyExitClearanceConstraint(Constraint):
    """Constraint to ensure emergency exits have adequate clearance"""
    
    def __init__(self, min_clearance: float):
        """
        Initialize emergency exit clearance constraint
        
        Args:
            min_clearance: Minimum clearance required around emergency exits (in layout units)
        """
        super().__init__(
            name="Emergency Exit Clearance",
            description=f"Ensures all emergency exits have at least {min_clearance} units of clearance",
            severity=ConstraintSeverity.CRITICAL
        )
        self.min_clearance = min_clearance
        
    def validate(self, layout) -> List[ConstraintViolation]:
        violations = []
        
        # Get all emergency exits and all other elements
        exits = layout.get_elements_by_type("emergency_exit")
        other_elements = [elem for elem in layout.all_elements() if elem.type != "emergency_exit"]
        
        for exit_elem in exits:
            exit_bounds = exit_elem.get_bounds().expand(self.min_clearance)
            
            for elem in other_elements:
                if exit_bounds.intersects(elem.get_bounds()):
                    violations.append(
                        ConstraintViolation(
                            message=f"Element {elem.id} intrudes on the required clearance around emergency exit {exit_elem.id}",
                            severity=self.severity,
                            location=exit_bounds,
                            elements_involved=[exit_elem.id, elem.id]
                        )
                    )
                    
        return violations


class FireSafetyConstraint(Constraint):
    """Constraint to ensure fire safety requirements are met"""
    
    def __init__(self, max_distance_to_exit: float, sprinkler_coverage_required: bool = True):
        """
        Initialize fire safety constraint
        
        Args:
            max_distance_to_exit: Maximum distance allowed from any point to an emergency exit
            sprinkler_coverage_required: Whether sprinkler coverage is required
        """
        super().__init__(
            name="Fire Safety Requirements",
            description="Ensures fire safety requirements are met",
            severity=ConstraintSeverity.CRITICAL
        )
        self.max_distance_to_exit = max_distance_to_exit
        self.sprinkler_coverage_required = sprinkler_coverage_required
        
    def validate(self, layout) -> List[ConstraintViolation]:
        violations = []
        
        # Get all emergency exits and all areas that need to be checked
        exits = layout.get_elements_by_type("emergency_exit")
        exit_points = [exit_elem.get_center() for exit_elem in exits]
        
        # Check distance from all walkable areas to nearest exit
        walkable_areas = layout.get_walkable_areas()
        sample_points = layout.get_sample_points(max_distance=5.0)  # Sample points every 5 units
        
        for point in sample_points:
            if not any(walkable_area.contains(point) for walkable_area in walkable_areas):
                continue
                
            nearest_exit_dist = min([calculate_distance(point, exit_point) for exit_point in exit_points], 
                                    default=float('inf'))
            
            if nearest_exit_dist > self.max_distance_to_exit:
                violations.append(
                    ConstraintViolation(
                        message=f"Point at {point} is {nearest_exit_dist:.2f} units from the nearest exit, "
                               f"exceeding the maximum of {self.max_distance_to_exit}",
                        severity=self.severity,
                        location=point
                    )
                )
        
        # Check sprinkler coverage if required
        if self.sprinkler_coverage_required:
            sprinklers = layout.get_elements_by_type("sprinkler")
            if not sprinklers:
                violations.append(
                    ConstraintViolation(
                        message="No sprinklers found in the layout",
                        severity=self.severity
                    )
                )
            else:
                # Additional sprinkler coverage validation could be implemented here
                pass
                
        return violations


class LoadingDockAccessConstraint(Constraint):
    """Constraint to ensure proper access to loading docks"""
    
    def __init__(self, min_clearance_width: float, min_maneuvering_area: float):
        """
        Initialize loading dock access constraint
        
        Args:
            min_clearance_width: Minimum width for loading dock access paths
            min_maneuvering_area: Minimum area needed for truck maneuvering
        """
        super().__init__(
            name="Loading Dock Access",
            description="Ensures loading docks have proper access paths and maneuvering space",
            severity=ConstraintSeverity.HIGH
        )
        self.min_clearance_width = min_clearance_width
        self.min_maneuvering_area = min_maneuvering_area
        
    def validate(self, layout) -> List[ConstraintViolation]:
        violations = []
        
        # Get all loading docks and access paths
        loading_docks = layout.get_elements_by_type("loading_dock")
        
        for dock in loading_docks:
            # Check for existence of access path
            access_path = layout.get_path_to_element(dock.id)
            if not access_path:
                violations.append(
                    ConstraintViolation(
                        message=f"No clear access path found to loading dock {dock.id}",
                        severity=self.severity,
                        location=dock.get_bounds(),
                        elements_involved=[dock.id]
                    )
                )
                continue
            
            # Check width of access path
            narrow_segments = [segment for segment in access_path.segments if segment.width < self.min_clearance_width]
            if narrow_segments:
                for segment in narrow_segments:
                    violations.append(
                        ConstraintViolation(
                            message=f"Access path to loading dock {dock.id} has segment with width {segment.width:.2f}, "
                                   f"less than required {self.min_clearance_width}",
                            severity=self.severity,
                            location=segment.get_bounds(),
                            elements_involved=[dock.id, segment.id]
                        )
                    )
            
            # Check maneuvering area
            maneuvering_area = layout.get_maneuvering_area(dock.id)
            if maneuvering_area < self.min_maneuvering_area:
                violations.append(
                    ConstraintViolation(
                        message=f"Insufficient maneuvering area ({maneuvering_area:.2f} sq units) for loading dock {dock.id}, "
                               f"minimum required is {self.min_maneuvering_area} sq units",
                        severity=self.severity,
                        location=dock.get_bounds(),
                        elements_involved=[dock.id]
                    )
                )
                
        return violations


class WorkflowEfficiencyConstraint(Constraint):
    """Constraint to optimize workflow efficiency in the layout"""
    
    def __init__(self, workflow_data: Dict[str, Dict[str, float]], max_flow_distance: float):
        """
        Initialize workflow efficiency constraint
        
        Args:
            workflow_data: Dictionary mapping pairs of element types to their flow volume
            max_flow_distance: Maximum weighted distance for efficient workflow
        """
        super().__init__(
            name="Workflow Efficiency",
            description="Ensures the layout supports efficient workflow between related areas",
            severity=ConstraintSeverity.MEDIUM
        )
        self.workflow_data = workflow_data
        self.max_flow_distance = max_flow_distance
        
    def validate(self, layout) -> List[ConstraintViolation]:
        violations = []
        
        # For each pair of elements with workflow between them
        for source_type, destinations in self.workflow_data.items():
            source_elements = layout.get_elements_by_type(source_type)
            
            for dest_type, flow_volume in destinations.items():
                dest_elements = layout.get_elements_by_type(dest_type)
                
                # Skip if no elements of these types exist
                if not source_elements or not dest_elements:
                    continue
                
                # Check the workflow distance between each pair
                for source in source_elements:
                    for dest in dest_elements:
                        path = layout.get_shortest_path(source.id, dest.id)
                        if not path:
                            violations.append(
                                ConstraintViolation(
                                    message=f"No path found between {source.id} and {dest.id}, "
                                           f"which have a workflow volume of {flow_volume}",
                                    severity=self.severity,
                                    elements_involved=[source.id, dest.id]
                                )
                            )
                            continue
                        
                        # Calculate weighted distance based on flow volume
                        weighted_distance = path.length * flow_volume
                        if weighted_distance > self.max_flow_distance:
                            violations.append(
                                ConstraintViolation(
                                    message=f"Inefficient workflow between {source.id} and {dest.id}: "
                                           f"weighted distance {weighted_distance:.2f} exceeds maximum {self.max_flow_distance}",
                                    severity=self.severity,
                                    location=path.get_bounds(),
                                    elements_involved=[source.id, dest.id]
                                )
                            )
                            
        return violations


class AisleIntersectionConstraint(Constraint):
    """Constraint to check for proper aisle intersections"""
    
    def __init__(self, min_corner_radius: float = 0.0):
        """
        Initialize aisle intersection constraint
        
        Args:
            min_corner_radius: Minimum radius for aisle corners for forklift turning
        """
        super().__init__(
            name="Aisle Intersection Design",
            description="Ensures aisle intersections are properly designed for traffic flow",
            severity=ConstraintSeverity.MEDIUM
        )
        self.min_corner_radius = min_corner_radius
        
    def validate(self, layout) -> List[ConstraintViolation]:
        violations = []
        
        # Get all aisle intersections
        intersections = layout.get_aisle_intersections()
        
        for intersection in intersections:
            if intersection.corner_radius < self.min_corner_radius:
                violations.append(
                    ConstraintViolation(
                        message=f"Aisle intersection at {intersection.location} has corner radius "
                               f"{intersection.corner_radius:.2f}, less than minimum {self.min_corner_radius}",
                        severity=self.severity,
                        location=intersection.get_bounds()
                    )
                )
                
            # Check for 4-way intersections without sufficient visibility
            if len(intersection.connecting_aisles) >= 4 and not intersection.has_visibility_aids:
                violations.append(
                    ConstraintViolation(
                        message=f"4-way intersection at {intersection.location} lacks visibility aids",
                        severity=ConstraintSeverity.LOW,
                        location=intersection.get_bounds()
                    )
                )
                
        return violations


class ConstraintEngine:
    """Engine for applying and validating all constraints on a layout"""
    
    def __init__(self):
        """Initialize the constraint engine"""
        self.constraints = []
        
    def add_constraint(self, constraint: Constraint):
        """Add a constraint to the engine"""
        self.constraints.append(constraint)
        return self
        
    def add_standard_constraints(self, config):
        """Add a standard set of constraints based on configuration"""
        # Add minimum aisle width constraint
        self.add_constraint(MinimumAisleWidthConstraint(
            min_width=config.get("minimum_aisle_width", 2.5),
            forklift_required=config.get("forklift_required", True)
        ))
        
        # Add emergency exit clearance constraint
        self.add_constraint(EmergencyExitClearanceConstraint(
            min_clearance=config.get("emergency_exit_clearance", 1.5)
        ))
        
        # Add fire safety constraint
        self.add_constraint(FireSafetyConstraint(
            max_distance_to_exit=config.get("max_distance_to_exit", 50.0),
            sprinkler_coverage_required=config.get("sprinkler_coverage_required", True)
        ))
        
        # Add loading dock access constraint if applicable
        if config.get("has_loading_docks", False):
            self.add_constraint(LoadingDockAccessConstraint(
                min_clearance_width=config.get("loading_dock_access_width", 4.0),
                min_maneuvering_area=config.get("loading_dock_maneuvering_area", 100.0)
            ))
        
        # Add workflow efficiency constraint if workflow data is available
        workflow_data = config.get("workflow_data")
        if workflow_data:
            self.add_constraint(WorkflowEfficiencyConstraint(
                workflow_data=workflow_data,
                max_flow_distance=config.get("max_flow_distance", 200.0)
            ))
            
        # Add aisle intersection constraint
        self.add_constraint(AisleIntersectionConstraint(
            min_corner_radius=config.get("aisle_corner_radius", 1.5)
        ))
        
        return self
        
    def validate_layout(self, layout) -> List[ConstraintViolation]:
        """
        Validate all constraints against the layout
        
        Args:
            layout: The warehouse layout to validate
            
        Returns:
            List of all constraint violations found
        """
        all_violations = []
        
        for constraint in self.constraints:
            violations = constraint.validate(layout)
            all_violations.extend(violations)
            
        # Sort violations by severity
        all_violations.sort(key=lambda v: v.severity.value, reverse=True)
        
        return all_violations
    
    def get_critical_violations(self, violations: List[ConstraintViolation]) -> List[ConstraintViolation]:
        """Get only critical violations from a list of violations"""
        return [v for v in violations if v.severity == ConstraintSeverity.CRITICAL]
    
    def has_critical_violations(self, layout) -> bool:
        """Check if the layout has any critical violations"""
        for constraint in self.constraints:
            violations = constraint.validate(layout)
            if any(v.severity == ConstraintSeverity.CRITICAL for v in violations):
                return True
        return False
    
    def generate_constraint_report(self, layout) -> Dict:
        """
        Generate a detailed report of constraint violations
        
        Args:
            layout: The warehouse layout to validate
            
        Returns:
            Dictionary with report data
        """
        all_violations = self.validate_layout(layout)
        
        # Group violations by severity
        violations_by_severity = {
            ConstraintSeverity.CRITICAL: [],
            ConstraintSeverity.HIGH: [],
            ConstraintSeverity.MEDIUM: [],
            ConstraintSeverity.LOW: []
        }
        
        for violation in all_violations:
            violations_by_severity[violation.severity].append(violation)
            
        # Create summary
        summary = {
            "total_violations": len(all_violations),
            "critical_violations": len(violations_by_severity[ConstraintSeverity.CRITICAL]),
            "high_violations": len(violations_by_severity[ConstraintSeverity.HIGH]),
            "medium_violations": len(violations_by_severity[ConstraintSeverity.MEDIUM]),
            "low_violations": len(violations_by_severity[ConstraintSeverity.LOW]),
            "is_valid": len(violations_by_severity[ConstraintSeverity.CRITICAL]) == 0
        }
        
        # Create detailed report
        report = {
            "summary": summary,
            "violations": violations_by_severity,
            "layout_stats": layout.get_stats()
        }
        
        return report