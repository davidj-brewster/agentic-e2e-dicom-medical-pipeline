"""
Measurement tools for neuroimaging visualization.
Implements distance, angle, and area/volume measurements.
"""
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Arc, Circle, Polygon
from matplotlib.text import Text

from utils.interaction import Point
from utils.measurement import (
    Coordinate,
    CoordinateSystem,
    MeasurementResult,
    MeasurementSystem,
    UnitSystem
)

logger = logging.getLogger(__name__)


@dataclass
class PreviewState:
    """Preview state for measurement tools"""
    points: List[Point]
    current_point: Optional[Point] = None
    active: bool = False
    complete: bool = False


class AngleMode(Enum):
    """Available angle measurement modes"""
    THREE_POINT = auto()    # Three-point angle
    TWO_LINE = auto()       # Two-line angle
    NORMAL = auto()         # Normal to plane
    ORIENTATION = auto()    # Orientation angle


class OrientationMode(Enum):
    """Available orientation modes"""
    CLOCKWISE = auto()          # Measure angle clockwise
    COUNTERCLOCKWISE = auto()   # Measure angle counterclockwise
    ABSOLUTE = auto()           # Measure absolute angle


class ROIMode(Enum):
    """Available ROI selection modes"""
    POLYGON = auto()     # Freehand polygon
    RECTANGLE = auto()   # Rectangular selection
    CIRCLE = auto()      # Circular selection
    SMART = auto()       # Smart selection


class ROIStatistics:
    """ROI statistical analysis"""
    
    def __init__(self, data: np.ndarray, mask: np.ndarray):
        """
        Initialize ROI statistics.
        
        Args:
            data: Image data
            mask: ROI mask
        """
        self.data = data
        self.mask = mask
    
    def calculate(self) -> Dict[str, float]:
        """
        Calculate ROI statistics.
        
        Returns:
            Dictionary of statistics
        """
        # Extract ROI data
        roi_data = self.data[self.mask > 0]
        
        if len(roi_data) == 0:
            return {
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "median": 0.0,
                "volume": 0.0,
                "area": 0.0
            }
        
        # Calculate statistics
        return {
            "mean": float(np.mean(roi_data)),
            "std": float(np.std(roi_data)),
            "min": float(np.min(roi_data)),
            "max": float(np.max(roi_data)),
            "median": float(np.median(roi_data)),
            "volume": float(np.sum(self.mask)),
            "area": float(self.calculate_surface_area())
        }
    
    def calculate_surface_area(self) -> float:
        """
        Calculate ROI surface area.
        
        Returns:
            Surface area in pixels
        """
        # Use marching squares to find contours
        from skimage import measure
        contours = measure.find_contours(self.mask, 0.5)
        
        if not contours:
            return 0.0
        
        # Calculate perimeter of largest contour
        largest_contour = max(contours, key=len)
        dx = np.diff(largest_contour[:, 1])
        dy = np.diff(largest_contour[:, 0])
        return float(np.sum(np.sqrt(dx*dx + dy*dy)))


class MeasurementTool(ABC):
    """Base class for measurement tools"""
    
    def __init__(self, system: MeasurementSystem):
        """
        Initialize measurement tool.
        
        Args:
            system: Measurement system
        """
        self.system = system
        self.preview = PreviewState(points=[])
        self.preview_artists: List[Artist] = []
        self.result: Optional[MeasurementResult] = None
    
    @abstractmethod
    def handle_click(self, point: Point) -> None:
        """
        Handle mouse click event.
        
        Args:
            point: Click point
        """
        pass
    
    @abstractmethod
    def handle_move(self, point: Point) -> None:
        """
        Handle mouse move event.
        
        Args:
            point: Current point
        """
        pass
    
    @abstractmethod
    def calculate(self) -> float:
        """
        Calculate measurement value.
        
        Returns:
            Measurement value
        """
        pass
    
    @abstractmethod
    def draw_preview(self, ax: Axes) -> None:
        """
        Draw measurement preview.
        
        Args:
            ax: Matplotlib axes
        """
        pass
    
    def clear_preview(self) -> None:
        """Clear preview artists"""
        for artist in self.preview_artists:
            artist.remove()
        self.preview_artists = []
    
    def reset(self) -> None:
        """Reset tool state"""
        self.preview = PreviewState(points=[])
        self.clear_preview()
        self.result = None
    
    def finish_measurement(self) -> None:
        """Finish measurement and store result"""
        value = self.calculate()
        if value > 0:
            self.result = MeasurementResult(
                value=value,
                unit=self.system.unit_system,
                points=[
                    Coordinate(p.x, p.y)
                    for p in self.preview.points
                ],
                type=self.__class__.__name__.lower()
            )
            self.system.add_result(self.result)
            logger.debug(f"Measurement complete: {value:.2f}")
        self.preview.complete = True


class DistanceTool(MeasurementTool):
    """Distance measurement tool"""
    
    def handle_click(self, point: Point) -> None:
        """Handle mouse click event"""
        if self.preview.complete:
            self.reset()
        
        self.preview.points.append(point)
        if len(self.preview.points) >= 2:
            self.finish_measurement()
    
    def handle_move(self, point: Point) -> None:
        """Handle mouse move event"""
        if not self.preview.complete:
            self.preview.current_point = point
    
    def calculate(self) -> float:
        """Calculate distance measurement"""
        if len(self.preview.points) < 2:
            return 0.0
        
        # Get points
        p1 = self.preview.points[0]
        p2 = self.preview.points[1]
        
        # Convert to physical coordinates
        c1 = self.system.convert_coordinates(
            Coordinate(p1.x, p1.y),
            CoordinateSystem.IMAGE,
            CoordinateSystem.PHYSICAL
        )
        c2 = self.system.convert_coordinates(
            Coordinate(p2.x, p2.y),
            CoordinateSystem.IMAGE,
            CoordinateSystem.PHYSICAL
        )
        
        # Calculate Euclidean distance
        distance = np.sqrt(
            (c2.x - c1.x)**2 +
            (c2.y - c1.y)**2 +
            (c2.z - c1.z)**2
        )
        
        # Convert to preferred unit system
        if self.system.unit_system != UnitSystem.MILLIMETERS:
            distance = self.system.convert_value(
                distance,
                UnitSystem.MILLIMETERS,
                self.system.unit_system
            )
        
        return distance
    
    def draw_preview(self, ax: Axes) -> None:
        """Draw distance measurement preview"""
        # Clear previous preview
        self.clear_preview()
        
        if not self.preview.points:
            return
        
        # Draw start point
        start = self.preview.points[0]
        point = Circle(
            (start.x, start.y),
            radius=3,
            color='red',
            fill=True,
            alpha=0.6
        )
        ax.add_artist(point)
        self.preview_artists.append(point)
        
        # Draw line to current point
        end = (
            self.preview.points[1]
            if len(self.preview.points) > 1
            else self.preview.current_point
        )
        if end:
            line = Line2D(
                [start.x, end.x],
                [start.y, end.y],
                color='red',
                linestyle='--',
                alpha=0.6
            )
            ax.add_artist(line)
            self.preview_artists.append(line)
            
            # Draw end point
            point = Circle(
                (end.x, end.y),
                radius=3,
                color='red',
                fill=True,
                alpha=0.6
            )
            ax.add_artist(point)
            self.preview_artists.append(point)
            
            # Draw distance label
            if len(self.preview.points) > 1:
                distance = self.calculate()
                mid_x = (start.x + end.x) / 2
                mid_y = (start.y + end.y) / 2
                label = Text(
                    mid_x, mid_y,
                    f"{distance:.1f} {self.system.unit_system.name}",
                    color='red',
                    backgroundcolor='white',
                    horizontalalignment='center',
                    verticalalignment='bottom'
                )
                ax.add_artist(label)
                self.preview_artists.append(label)
        
        # Request redraw
        ax.figure.canvas.draw_idle()


class AngleTool(MeasurementTool):
    """Angle measurement tool"""
    
    def __init__(self, system: MeasurementSystem):
        """Initialize angle tool"""
        super().__init__(system)
        self.mode = AngleMode.THREE_POINT
        self.orientation = OrientationMode.CLOCKWISE
    
    def handle_click(self, point: Point) -> None:
        """Handle mouse click event"""
        if self.preview.complete:
            self.reset()
        
        self.preview.points.append(point)
        if len(self.preview.points) >= 3:
            self.finish_measurement()
    
    def handle_move(self, point: Point) -> None:
        """Handle mouse move event"""
        if not self.preview.complete:
            self.preview.current_point = point
    
    def calculate(self) -> float:
        """Calculate angle measurement"""
        if len(self.preview.points) < 3:
            return 0.0
        
        # Get vectors
        p1 = self.preview.points[0]
        p2 = self.preview.points[1]
        p3 = self.preview.points[2]
        
        # Convert to physical coordinates
        c1 = self.system.convert_coordinates(
            Coordinate(p1.x, p1.y),
            CoordinateSystem.IMAGE,
            CoordinateSystem.PHYSICAL
        )
        c2 = self.system.convert_coordinates(
            Coordinate(p2.x, p2.y),
            CoordinateSystem.IMAGE,
            CoordinateSystem.PHYSICAL
        )
        c3 = self.system.convert_coordinates(
            Coordinate(p3.x, p3.y),
            CoordinateSystem.IMAGE,
            CoordinateSystem.PHYSICAL
        )
        
        # Calculate vectors
        v1 = np.array([c1.x - c2.x, c1.y - c2.y, c1.z - c2.z])
        v2 = np.array([c3.x - c2.x, c3.y - c2.y, c3.z - c2.z])
        
        # Calculate angle
        dot = np.dot(v1, v2)
        norms = np.linalg.norm(v1) * np.linalg.norm(v2)
        
        if norms == 0:
            return 0.0
        
        # Ensure dot product is in valid range [-1, 1]
        dot = np.clip(dot / norms, -1.0, 1.0)
        angle = np.arccos(dot)
        
        # Handle orientation
        if self.orientation != OrientationMode.ABSOLUTE:
            # Calculate cross product for 2D orientation
            cross = np.cross(v1, v2)
            if (self.orientation == OrientationMode.CLOCKWISE) == (cross[2] > 0):
                angle = 2 * np.pi - angle
        
        # Convert to degrees
        return np.degrees(angle)
    
    def draw_preview(self, ax: Axes) -> None:
        """Draw angle measurement preview"""
        # Clear previous preview
        self.clear_preview()
        
        if not self.preview.points:
            return
        
        # Draw points and lines
        colors = ['blue', 'blue', 'blue']
        points = (
            self.preview.points +
            ([self.preview.current_point] if self.preview.current_point else [])
        )
        
        # Draw points
        for i, point in enumerate(points):
            marker = Circle(
                (point.x, point.y),
                radius=3,
                color=colors[i],
                fill=True,
                alpha=0.6
            )
            ax.add_artist(marker)
            self.preview_artists.append(marker)
        
        # Draw lines
        if len(points) >= 2:
            line1 = Line2D(
                [points[0].x, points[1].x],
                [points[0].y, points[1].y],
                color='blue',
                linestyle='--',
                alpha=0.6
            )
            ax.add_artist(line1)
            self.preview_artists.append(line1)
        
        if len(points) >= 3:
            line2 = Line2D(
                [points[1].x, points[2].x],
                [points[1].y, points[2].y],
                color='blue',
                linestyle='--',
                alpha=0.6
            )
            ax.add_artist(line2)
            self.preview_artists.append(line2)
            
            # Calculate angle for arc
            dx1 = points[0].x - points[1].x
            dy1 = points[0].y - points[1].y
            dx2 = points[2].x - points[1].x
            dy2 = points[2].y - points[1].y
            
            angle1 = np.degrees(np.arctan2(dy1, dx1))
            angle2 = np.degrees(np.arctan2(dy2, dx2))
            
            # Ensure correct arc direction
            if self.orientation == OrientationMode.CLOCKWISE:
                if angle2 > angle1:
                    angle2 -= 360
            else:
                if angle1 > angle2:
                    angle1 -= 360
            
            # Draw angle arc
            arc = Arc(
                (points[1].x, points[1].y),
                width=20,
                height=20,
                angle=0,
                theta1=angle1,
                theta2=angle2,
                color='blue',
                alpha=0.6
            )
            ax.add_artist(arc)
            self.preview_artists.append(arc)
            
            # Draw angle label
            if len(self.preview.points) >= 3:
                angle = self.calculate()
                mid_angle = np.radians((angle1 + angle2) / 2)
                label = Text(
                    points[1].x + 25 * np.cos(mid_angle),
                    points[1].y + 25 * np.sin(mid_angle),
                    f"{angle:.1f}°",
                    color='blue',
                    backgroundcolor='white',
                    horizontalalignment='center',
                    verticalalignment='center'
                )
                ax.add_artist(label)
                self.preview_artists.append(label)
        
        # Request redraw
        ax.figure.canvas.draw_idle()


class AreaVolumeTool(MeasurementTool):
    """Area and volume measurement tool"""
    
    def __init__(self, system: MeasurementSystem):
        """Initialize area/volume tool"""
        super().__init__(system)
        self.mode = ROIMode.POLYGON
        self.closed = False
        self.mask: Optional[np.ndarray] = None
        self.stats: Optional[Dict[str, float]] = None
    
    def handle_click(self, point: Point) -> None:
        """Handle mouse click event"""
        if self.preview.complete:
            self.reset()
            self.closed = False
            self.mask = None
            self.stats = None
        
        # Close polygon on double click near start point
        if (len(self.preview.points) >= 3 and
            np.hypot(point.x - self.preview.points[0].x,
                    point.y - self.preview.points[0].y) < 5):
            self.closed = True
            self.finish_measurement()
            return
        
        self.preview.points.append(point)
    
    def handle_move(self, point: Point) -> None:
        """Handle mouse move event"""
        if not self.preview.complete:
            self.preview.current_point = point
    
    def calculate(self) -> Dict[str, float]:
        """Calculate area and volume measurements"""
        if len(self.preview.points) < 3 or not self.closed:
            return {"area": 0.0, "volume": 0.0}
        
        # Get points in physical coordinates
        coords = [
            self.system.convert_coordinates(
                Coordinate(p.x, p.y),
                CoordinateSystem.IMAGE,
                CoordinateSystem.PHYSICAL
            )
            for p in self.preview.points
        ]
        
        # Calculate area using shoelace formula
        x = [c.x for c in coords]
        y = [c.y for c in coords]
        area = 0.5 * abs(
            sum(i * j for i, j in zip(x, y[1:] + [y[0]])) -
            sum(i * j for i, j in zip(x[1:] + [x[0]], y))
        )
        
        # Calculate volume if mask available
        volume = 0.0
        if self.mask is not None:
            roi = self.create_roi_mask()
            volume = np.sum(self.mask * roi) * np.prod(
                self.system.voxel_size
            )
        
        # Convert units if needed
        if self.system.unit_system != UnitSystem.MILLIMETERS:
            area = self.system.convert_value(
                area,
                UnitSystem.MILLIMETERS,
                self.system.unit_system
            )
            volume = self.system.convert_value(
                volume,
                UnitSystem.MILLIMETERS,
                self.system.unit_system
            )
        
        return {
            "area": area,
            "volume": volume
        }
    
    def create_roi_mask(self) -> np.ndarray:
        """Create ROI mask from polygon points"""
        from skimage import draw
        
        # Create empty mask
        shape = self.mask.shape if self.mask is not None else (512, 512)
        mask = np.zeros(shape, dtype=bool)
        
        # Convert points to pixel coordinates
        vertices = np.array([
            [p.x, p.y] for p in self.preview.points
        ])
        
        # Create polygon mask
        rr, cc = draw.polygon(
            vertices[:, 1],
            vertices[:, 0],
            shape=mask.shape
        )
        mask[rr, cc] = True
        
        return mask
    
    def calculate_statistics(self) -> Dict[str, float]:
        """Calculate ROI statistics"""
        if not self.closed or self.mask is None:
            return {}
        
        # Create ROI mask
        roi = self.create_roi_mask()
        
        # Calculate statistics
        stats = ROIStatistics(self.mask, roi)
        self.stats = stats.calculate()
        
        return self.stats
    
    def draw_preview(self, ax: Axes) -> None:
        """Draw ROI preview"""
        # Clear previous preview
        self.clear_preview()
        
        if not self.preview.points:
            return
        
        # Draw vertices
        for point in self.preview.points:
            marker = Circle(
                (point.x, point.y),
                radius=3,
                color='green',
                fill=True,
                alpha=0.6
            )
            ax.add_artist(marker)
            self.preview_artists.append(marker)
        
        # Get all points including current
        points = (
            self.preview.points +
            ([self.preview.current_point] if self.preview.current_point else [])
        )
        
        # Draw polygon edges
        if len(points) >= 2:
            vertices = np.array([
                [p.x, p.y] for p in points
            ])
            if self.closed:
                vertices = np.vstack([vertices, vertices[0]])
            
            lines = Line2D(
                vertices[:, 0],
                vertices[:, 1],
                color='green',
                linestyle='--',
                alpha=0.6
            )
            ax.add_artist(lines)
            self.preview_artists.append(lines)
        
        # Draw ROI overlay and measurements
        if self.closed:
            # Create and draw ROI mask
            roi = self.create_roi_mask()
            mask_artist = ax.imshow(
                roi,
                cmap='Greens',
                alpha=0.3
            )
            self.preview_artists.append(mask_artist)
            
            # Calculate measurements
            measurements = self.calculate()
            stats = self.calculate_statistics()
            
            # Create label text
            label_text = [
                f"Area: {measurements['area']:.1f} {self.system.unit_system.name}²",
                f"Volume: {measurements['volume']:.1f} {self.system.unit_system.name}³"
            ]
            if stats:
                label_text.extend([
                    f"Mean: {stats['mean']:.1f}",
                    f"Std: {stats['std']:.1f}",
                    f"Min: {stats['min']:.1f}",
                    f"Max: {stats['max']:.1f}"
                ])
            
            # Add label
            label = Text(
                self.preview.points[0].x,
                self.preview.points[0].y - 10,
                "\n".join(label_text),
                color='green',
                backgroundcolor='white',
                horizontalalignment='left',
                verticalalignment='bottom'
            )
            ax.add_artist(label)
            self.preview_artists.append(label)
        
        # Request redraw
        ax.figure.canvas.draw_idle()