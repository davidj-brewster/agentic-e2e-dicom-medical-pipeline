"""
Interactive visualization controls.
Handles user interaction with visualizations.
"""
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import vtk
from matplotlib.backend_bases import MouseEvent, KeyEvent
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)


class InteractionMode(Enum):
    """Available interaction modes"""
    PAN = auto()
    ZOOM = auto()
    WINDOW_LEVEL = auto()
    CROSSHAIR = auto()
    NONE = auto()


@dataclass
class Point:
    """2D point coordinates"""
    x: float
    y: float


@dataclass
class ViewState:
    """Current view state"""
    center: Point
    zoom: float
    window: float
    level: float
    orientation: str


class InteractionHandler(ABC):
    """Base class for interaction handlers"""
    
    def __init__(self):
        self.active = False
        self.start_point: Optional[Point] = None
        self.current_point: Optional[Point] = None
        self.view_state: Optional[ViewState] = None
    
    @abstractmethod
    def on_mouse_press(self, event: MouseEvent) -> None:
        """Handle mouse press event"""
        pass
    
    @abstractmethod
    def on_mouse_move(self, event: MouseEvent) -> None:
        """Handle mouse move event"""
        pass
    
    @abstractmethod
    def on_mouse_release(self, event: MouseEvent) -> None:
        """Handle mouse release event"""
        pass
    
    @abstractmethod
    def on_key_press(self, event: KeyEvent) -> None:
        """Handle key press event"""
        pass


class PanHandler(InteractionHandler):
    """Pan interaction handler"""
    
    def on_mouse_press(self, event: MouseEvent) -> None:
        """Handle mouse press event"""
        if event.button == 1:  # Left click
            self.active = True
            self.start_point = Point(event.xdata, event.ydata)
            self.current_point = self.start_point
    
    def on_mouse_move(self, event: MouseEvent) -> None:
        """Handle mouse move event"""
        if self.active and event.xdata is not None and event.ydata is not None:
            # Calculate pan delta
            dx = event.xdata - self.current_point.x
            dy = event.ydata - self.current_point.y
            
            # Update view center
            self.view_state.center.x -= dx
            self.view_state.center.y -= dy
            
            # Update current point
            self.current_point = Point(event.xdata, event.ydata)
            
            # Request redraw
            event.canvas.draw_idle()
    
    def on_mouse_release(self, event: MouseEvent) -> None:
        """Handle mouse release event"""
        if event.button == 1:  # Left click
            self.active = False
            self.start_point = None
            self.current_point = None
    
    def on_key_press(self, event: KeyEvent) -> None:
        """Handle key press event"""
        pass


class ZoomHandler(InteractionHandler):
    """Zoom interaction handler"""
    
    def on_mouse_press(self, event: MouseEvent) -> None:
        """Handle mouse press event"""
        if event.button == 3:  # Right click
            self.active = True
            self.start_point = Point(event.xdata, event.ydata)
            self.current_point = self.start_point
    
    def on_mouse_move(self, event: MouseEvent) -> None:
        """Handle mouse move event"""
        if self.active and event.ydata is not None:
            # Calculate zoom factor based on vertical movement
            zoom_factor = 1.0 + (event.ydata - self.current_point.y) / 100.0
            
            # Update zoom level
            self.view_state.zoom *= zoom_factor
            
            # Update current point
            self.current_point = Point(event.xdata, event.ydata)
            
            # Request redraw
            event.canvas.draw_idle()
    
    def on_mouse_release(self, event: MouseEvent) -> None:
        """Handle mouse release event"""
        if event.button == 3:  # Right click
            self.active = False
            self.start_point = None
            self.current_point = None
    
    def on_key_press(self, event: KeyEvent) -> None:
        """Handle key press event"""
        if event.key in ['+', '=']:
            self.view_state.zoom *= 1.1
            event.canvas.draw_idle()
        elif event.key in ['-', '_']:
            self.view_state.zoom /= 1.1
            event.canvas.draw_idle()


class WindowLevelHandler(InteractionHandler):
    """Window/level interaction handler"""
    
    def on_mouse_press(self, event: MouseEvent) -> None:
        """Handle mouse press event"""
        if event.button == 2:  # Middle click
            self.active = True
            self.start_point = Point(event.xdata, event.ydata)
            self.current_point = self.start_point
    
    def on_mouse_move(self, event: MouseEvent) -> None:
        """Handle mouse move event"""
        if self.active and event.xdata is not None and event.ydata is not None:
            # Calculate window/level adjustments
            dw = (event.xdata - self.current_point.x) * 10.0
            dl = (event.ydata - self.current_point.y) * 10.0
            
            # Update window/level
            self.view_state.window = max(1.0, self.view_state.window + dw)
            self.view_state.level += dl
            
            # Update current point
            self.current_point = Point(event.xdata, event.ydata)
            
            # Request redraw
            event.canvas.draw_idle()
    
    def on_mouse_release(self, event: MouseEvent) -> None:
        """Handle mouse release event"""
        if event.button == 2:  # Middle click
            self.active = False
            self.start_point = None
            self.current_point = None
    
    def on_key_press(self, event: KeyEvent) -> None:
        """Handle key press event"""
        pass


class CrosshairHandler(InteractionHandler):
    """Crosshair navigation handler"""
    
    def on_mouse_press(self, event: MouseEvent) -> None:
        """Handle mouse press event"""
        if event.button == 1 and event.dblclick:  # Double left click
            self.active = not self.active
            if self.active:
                self.current_point = Point(event.xdata, event.ydata)
                event.canvas.draw_idle()
    
    def on_mouse_move(self, event: MouseEvent) -> None:
        """Handle mouse move event"""
        if self.active and event.xdata is not None and event.ydata is not None:
            self.current_point = Point(event.xdata, event.ydata)
            event.canvas.draw_idle()
    
    def on_mouse_release(self, event: MouseEvent) -> None:
        """Handle mouse release event"""
        pass
    
    def on_key_press(self, event: KeyEvent) -> None:
        """Handle key press event"""
        if not self.active:
            return
            
        if event.key == 'up':
            self.current_point.y += 1
        elif event.key == 'down':
            self.current_point.y -= 1
        elif event.key == 'left':
            self.current_point.x -= 1
        elif event.key == 'right':
            self.current_point.x += 1
        
        event.canvas.draw_idle()


class InteractionManager:
    """Manages visualization interaction"""
    
    def __init__(self):
        self.mode = InteractionMode.NONE
        self.handlers = {
            InteractionMode.PAN: PanHandler(),
            InteractionMode.ZOOM: ZoomHandler(),
            InteractionMode.WINDOW_LEVEL: WindowLevelHandler(),
            InteractionMode.CROSSHAIR: CrosshairHandler()
        }
        self.view_state = ViewState(
            center=Point(0, 0),
            zoom=1.0,
            window=1.0,
            level=0.5,
            orientation="axial"
        )
        
        # Initialize handlers
        for handler in self.handlers.values():
            handler.view_state = self.view_state
    
    def set_mode(self, mode: InteractionMode) -> None:
        """Set interaction mode"""
        if self.mode != mode:
            # Deactivate current handler
            if self.mode != InteractionMode.NONE:
                self.handlers[self.mode].active = False
            
            self.mode = mode
            logger.debug(f"Interaction mode set to: {mode.name}")
    
    def handle_mouse_press(self, event: MouseEvent) -> None:
        """Handle mouse press event"""
        if self.mode != InteractionMode.NONE:
            self.handlers[self.mode].on_mouse_press(event)
    
    def handle_mouse_move(self, event: MouseEvent) -> None:
        """Handle mouse move event"""
        if self.mode != InteractionMode.NONE:
            self.handlers[self.mode].on_mouse_move(event)
    
    def handle_mouse_release(self, event: MouseEvent) -> None:
        """Handle mouse release event"""
        if self.mode != InteractionMode.NONE:
            self.handlers[self.mode].on_mouse_release(event)
    
    def handle_key_press(self, event: KeyEvent) -> None:
        """Handle key press event"""
        # Global key bindings
        if event.key == 'p':
            self.set_mode(InteractionMode.PAN)
        elif event.key == 'z':
            self.set_mode(InteractionMode.ZOOM)
        elif event.key == 'w':
            self.set_mode(InteractionMode.WINDOW_LEVEL)
        elif event.key == 'c':
            self.set_mode(InteractionMode.CROSSHAIR)
        elif event.key == 'escape':
            self.set_mode(InteractionMode.NONE)
        
        # Mode-specific key bindings
        if self.mode != InteractionMode.NONE:
            self.handlers[self.mode].on_key_press(event)


def connect_interaction_manager(fig: Figure, manager: InteractionManager) -> None:
    """Connect interaction manager to matplotlib figure"""
    fig.canvas.mpl_connect('button_press_event', manager.handle_mouse_press)
    fig.canvas.mpl_connect('motion_notify_event', manager.handle_mouse_move)
    fig.canvas.mpl_connect('button_release_event', manager.handle_mouse_release)
    fig.canvas.mpl_connect('key_press_event', manager.handle_key_press)