"""
Interactive 3D viewer with measurement tools.
Integrates VTK rendering with measurement tools.
"""
import logging
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import vtk
from matplotlib.figure import Figure
from vtk.util import numpy_support

from utils.measurement_tools import (
    AngleTool,
    AreaVolumeTool,
    DistanceTool,
    MeasurementTool
)

logger = logging.getLogger(__name__)


class ViewerMode(Enum):
    """Available viewer modes"""
    NAVIGATE = auto()    # Camera navigation
    MEASURE = auto()     # Measurement tools
    ANNOTATE = auto()    # Annotations


@dataclass
class ViewerConfig:
    """Viewer configuration"""
    width: int = 800
    height: int = 600
    background_color: Tuple[float, float, float] = (0.1, 0.1, 0.1)
    tool_color: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    tool_opacity: float = 0.8
    tool_line_width: float = 2.0


class Interactive3DViewer:
    """Interactive 3D viewer with measurement tools"""
    
    def __init__(
        self,
        config: ViewerConfig = ViewerConfig()
    ):
        """
        Initialize viewer.
        
        Args:
            config: Viewer configuration
        """
        self.config = config
        self.mode = ViewerMode.NAVIGATE
        self.tools: Dict[str, MeasurementTool] = {}
        self.active_tool: Optional[MeasurementTool] = None
        self.results: List[Dict] = []
        
        # Create VTK objects
        self.renderer = vtk.vtkRenderer()
        self.window = vtk.vtkRenderWindow()
        self.interactor = vtk.vtkRenderWindowInteractor()
        
        # Setup renderer
        self.renderer.SetBackground(*self.config.background_color)
        
        # Setup window
        self.window.SetSize(self.config.width, self.config.height)
        self.window.AddRenderer(self.renderer)
        
        # Setup interactor
        self.interactor.SetRenderWindow(self.window)
        self.interactor.SetInteractorStyle(
            vtk.vtkInteractorStyleTrackballCamera()
        )
        
        # Add observers
        self.interactor.AddObserver("LeftButtonPressEvent", self.on_left_click)
        self.interactor.AddObserver("MouseMoveEvent", self.on_mouse_move)
        self.interactor.AddObserver("KeyPressEvent", self.on_key_press)
    
    def add_tool(self, tool: MeasurementTool) -> None:
        """
        Add measurement tool.
        
        Args:
            tool: Measurement tool to add
        """
        tool_type = tool.__class__.__name__.lower()
        self.tools[tool_type] = tool
        
        # Create tool actors
        tool.actors = {
            "points": vtk.vtkActor(),
            "lines": vtk.vtkActor(),
            "labels": vtk.vtkActor()
        }
        
        # Setup tool properties
        for actor in tool.actors.values():
            actor.GetProperty().SetColor(*self.config.tool_color)
            actor.GetProperty().SetOpacity(self.config.tool_opacity)
            actor.GetProperty().SetLineWidth(self.config.tool_line_width)
            self.renderer.AddActor(actor)
        
        logger.debug(f"Added tool: {tool_type}")
    
    def switch_tool(self, tool_type: str) -> None:
        """
        Switch active measurement tool.
        
        Args:
            tool_type: Tool type to switch to
        """
        # Deactivate current tool
        if self.active_tool:
            self.active_tool.reset()
            for actor in self.active_tool.actors.values():
                actor.SetVisibility(False)
        
        # Activate new tool
        self.active_tool = self.tools.get(tool_type.lower())
        if self.active_tool:
            self.mode = ViewerMode.MEASURE
            for actor in self.active_tool.actors.values():
                actor.SetVisibility(True)
            logger.debug(f"Switched to tool: {tool_type}")
        else:
            self.mode = ViewerMode.NAVIGATE
            logger.warning(f"Tool not found: {tool_type}")
        
        self.window.Render()
    
    def on_left_click(self, obj: vtk.vtkObject, event: str) -> None:
        """
        Handle left click event.
        
        Args:
            obj: VTK object
            event: Event name
        """
        if self.mode != ViewerMode.MEASURE or not self.active_tool:
            return
        
        # Get click position
        click_pos = self.interactor.GetEventPosition()
        world_pos = self.get_world_position(click_pos)
        
        if world_pos is not None:
            # Handle click in active tool
            self.active_tool.handle_click(world_pos)
            self.update_tool_visualization()
            
            # Check if measurement is complete
            if self.active_tool.result:
                self.results.append({
                    "tool": self.active_tool.__class__.__name__,
                    "result": self.active_tool.result
                })
                logger.debug("Measurement complete")
    
    def on_mouse_move(self, obj: vtk.vtkObject, event: str) -> None:
        """
        Handle mouse move event.
        
        Args:
            obj: VTK object
            event: Event name
        """
        if self.mode != ViewerMode.MEASURE or not self.active_tool:
            return
        
        # Get cursor position
        cursor_pos = self.interactor.GetEventPosition()
        world_pos = self.get_world_position(cursor_pos)
        
        if world_pos is not None:
            # Update tool preview
            self.active_tool.handle_move(world_pos)
            self.update_tool_visualization()
    
    def on_key_press(self, obj: vtk.vtkObject, event: str) -> None:
        """
        Handle key press event.
        
        Args:
            obj: VTK object
            event: Event name
        """
        key = self.interactor.GetKeySym()
        
        if key == "Escape":
            # Reset active tool
            if self.active_tool:
                self.active_tool.reset()
                self.update_tool_visualization()
        
        elif key == "n":
            # Switch to navigation mode
            self.mode = ViewerMode.NAVIGATE
            if self.active_tool:
                for actor in self.active_tool.actors.values():
                    actor.SetVisibility(False)
            logger.debug("Switched to navigation mode")
        
        elif key == "m":
            # Switch to measurement mode
            self.mode = ViewerMode.MEASURE
            if self.active_tool:
                for actor in self.active_tool.actors.values():
                    actor.SetVisibility(True)
            logger.debug("Switched to measurement mode")
        
        elif key == "s":
            # Save screenshot
            self.save_screenshot()
        
        self.window.Render()
    
    def get_world_position(self, screen_pos: Tuple[int, int]) -> Optional[np.ndarray]:
        """
        Convert screen position to world position.
        
        Args:
            screen_pos: Screen coordinates (x, y)
            
        Returns:
            World coordinates (x, y, z) or None if no intersection
        """
        # Create picker
        picker = vtk.vtkPropPicker()
        picker.Pick(screen_pos[0], screen_pos[1], 0, self.renderer)
        
        # Get pick position
        world_pos = picker.GetPickPosition()
        if world_pos == (0, 0, 0):
            return None
        
        return np.array(world_pos)
    
    def update_tool_visualization(self) -> None:
        """Update tool visualization"""
        if not self.active_tool:
            return
        
        # Update points
        points = vtk.vtkPoints()
        for point in self.active_tool.preview.points:
            points.InsertNextPoint(point)
        
        # Create point data
        point_data = vtk.vtkPolyData()
        point_data.SetPoints(points)
        
        # Create point mapper
        point_mapper = vtk.vtkPolyDataMapper()
        point_mapper.SetInputData(point_data)
        
        # Update point actor
        self.active_tool.actors["points"].SetMapper(point_mapper)
        
        # Update lines if multiple points
        if len(self.active_tool.preview.points) >= 2:
            # Create lines
            lines = vtk.vtkCellArray()
            for i in range(len(self.active_tool.preview.points) - 1):
                line = vtk.vtkLine()
                line.GetPointIds().SetId(0, i)
                line.GetPointIds().SetId(1, i + 1)
                lines.InsertNextCell(line)
            
            # Create line data
            line_data = vtk.vtkPolyData()
            line_data.SetPoints(points)
            line_data.SetLines(lines)
            
            # Create line mapper
            line_mapper = vtk.vtkPolyDataMapper()
            line_mapper.SetInputData(line_data)
            
            # Update line actor
            self.active_tool.actors["lines"].SetMapper(line_mapper)
        
        # Update measurement label
        if self.active_tool.result:
            # Create label
            label = vtk.vtkVectorText()
            label.SetText(f"{self.active_tool.result.value:.1f}")
            
            # Create label mapper
            label_mapper = vtk.vtkPolyDataMapper()
            label_mapper.SetInputConnection(label.GetOutputPort())
            
            # Update label actor
            self.active_tool.actors["labels"].SetMapper(label_mapper)
            
            # Position label at measurement center
            center = np.mean(
                [p.to_array() for p in self.active_tool.result.points],
                axis=0
            )
            self.active_tool.actors["labels"].SetPosition(center)
        
        self.window.Render()
    
    def save_screenshot(self, path: Optional[Path] = None) -> None:
        """
        Save viewer screenshot.
        
        Args:
            path: Output file path
        """
        if not path:
            path = Path("screenshot.png")
        
        # Create window to image filter
        w2i = vtk.vtkWindowToImageFilter()
        w2i.SetInput(self.window)
        w2i.Update()
        
        # Create PNG writer
        writer = vtk.vtkPNGWriter()
        writer.SetFileName(str(path))
        writer.SetInputConnection(w2i.GetOutputPort())
        writer.Write()
        
        logger.info(f"Saved screenshot to: {path}")
    
    def start(self) -> None:
        """Start viewer"""
        self.window.Render()
        self.interactor.Initialize()
        self.interactor.Start()
    
    def stop(self) -> None:
        """Stop viewer"""
        self.interactor.TerminateApp()
        self.window.Finalize()
        del self.window