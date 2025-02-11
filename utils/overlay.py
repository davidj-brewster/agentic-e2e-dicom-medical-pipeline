"""
Overlay visualization functionality.
Handles image overlay visualization and analysis.
"""
import logging
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from skimage import measure

logger = logging.getLogger(__name__)


class OverlayMode(Enum):
    """Available overlay modes"""
    BLEND = auto()          # Alpha blending
    CHECKERBOARD = auto()   # Checkerboard pattern
    DIFFERENCE = auto()     # Difference map
    SPLIT = auto()          # Split view
    SIDE_BY_SIDE = auto()   # Side by side view


@dataclass
class VisualizationConfig:
    """Visualization configuration"""
    window: int = 1000              # Window width
    level: int = 0                  # Window level
    blend_factor: float = 0.5       # Blend factor
    checkerboard_size: int = 32     # Checkerboard block size
    histogram_bins: int = 100       # Number of histogram bins
    split_position: float = 0.5     # Split view position
    colormap: str = "gray"          # Base colormap
    overlay_colormap: str = "hot"   # Overlay colormap
    overlay_alpha: float = 0.5      # Overlay opacity


class OverlayVisualizer:
    """Overlay visualization system"""
    
    def __init__(
        self,
        base_image: np.ndarray,
        overlay_image: np.ndarray,
        config: Optional[VisualizationConfig] = None
    ):
        """
        Initialize visualizer.
        
        Args:
            base_image: Base image
            overlay_image: Overlay image
            config: Visualization configuration
        """
        self.base = base_image
        self.overlay = overlay_image
        self.config = config or VisualizationConfig()
        self.mode = OverlayMode.BLEND
        self.difference_map = None
        self.metrics = {}
        
        # Validate images
        if self.base.shape != self.overlay.shape:
            raise ValueError("Base and overlay images must have same shape")
    
    def normalize_image(self, data: np.ndarray) -> np.ndarray:
        """
        Normalize image data.
        
        Args:
            data: Input image data
            
        Returns:
            Normalized image data
        """
        # Apply window/level
        min_val = self.config.level - self.config.window/2
        max_val = self.config.level + self.config.window/2
        
        # Clip and normalize
        normalized = np.clip(data, min_val, max_val)
        normalized = (normalized - min_val) / self.config.window
        
        return normalized
    
    def render_blend(self) -> np.ndarray:
        """
        Render blended overlay.
        
        Returns:
            Blended image
        """
        base_norm = self.normalize_image(self.base)
        overlay_norm = self.normalize_image(self.overlay)
        
        return (
            (1 - self.config.blend_factor) * base_norm +
            self.config.blend_factor * overlay_norm
        )
    
    def render_checkerboard(self) -> np.ndarray:
        """
        Render checkerboard pattern.
        
        Returns:
            Checkerboard image
        """
        base_norm = self.normalize_image(self.base)
        overlay_norm = self.normalize_image(self.overlay)
        
        # Create checkerboard mask
        x, y = np.indices(base_norm.shape)
        block_size = self.config.checkerboard_size
        mask = ((x//block_size + y//block_size) % 2).astype(bool)
        
        result = np.where(mask, base_norm, overlay_norm)
        return result
    
    def render_difference(self) -> np.ndarray:
        """
        Render difference map.
        
        Returns:
            Difference map
        """
        base_norm = self.normalize_image(self.base)
        overlay_norm = self.normalize_image(self.overlay)
        
        self.difference_map = np.abs(base_norm - overlay_norm)
        return self.difference_map
    
    def render_split(self) -> np.ndarray:
        """
        Render split view.
        
        Returns:
            Split view image
        """
        base_norm = self.normalize_image(self.base)
        overlay_norm = self.normalize_image(self.overlay)
        
        # Create split mask
        split_idx = int(base_norm.shape[1] * self.config.split_position)
        mask = np.zeros_like(base_norm, dtype=bool)
        mask[:, :split_idx] = True
        
        result = np.where(mask, base_norm, overlay_norm)
        return result
    
    def render_side_by_side(self) -> Figure:
        """
        Render side by side view.
        
        Returns:
            Matplotlib figure
        """
        base_norm = self.normalize_image(self.base)
        overlay_norm = self.normalize_image(self.overlay)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Plot base image
        ax1.imshow(
            base_norm,
            cmap=self.config.colormap
        )
        ax1.set_title("Base Image")
        ax1.axis('off')
        
        # Plot overlay image
        ax2.imshow(
            overlay_norm,
            cmap=self.config.colormap
        )
        ax2.set_title("Overlay Image")
        ax2.axis('off')
        
        return fig
    
    def render(self) -> Union[np.ndarray, Figure]:
        """
        Render visualization.
        
        Returns:
            Rendered visualization
        """
        if self.mode == OverlayMode.BLEND:
            return self.render_blend()
        elif self.mode == OverlayMode.CHECKERBOARD:
            return self.render_checkerboard()
        elif self.mode == OverlayMode.DIFFERENCE:
            return self.render_difference()
        elif self.mode == OverlayMode.SPLIT:
            return self.render_split()
        else:  # SIDE_BY_SIDE
            return self.render_side_by_side()
    
    def calculate_metrics(self) -> Dict[str, float]:
        """
        Calculate comparison metrics.
        
        Returns:
            Dictionary of metrics
        """
        base_norm = self.normalize_image(self.base)
        overlay_norm = self.normalize_image(self.overlay)
        
        # Calculate difference map
        self.difference_map = np.abs(base_norm - overlay_norm)
        
        # Calculate metrics
        self.metrics = {
            "mse": float(np.mean(self.difference_map**2)),
            "mae": float(np.mean(self.difference_map)),
            "correlation": float(np.corrcoef(
                base_norm.ravel(),
                overlay_norm.ravel()
            )[0, 1])
        }
        
        return self.metrics
    
    def calculate_histogram(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate intensity histograms.
        
        Returns:
            Base and overlay histograms
        """
        # Calculate histograms
        base_hist, _ = np.histogram(
            self.base,
            bins=self.config.histogram_bins,
            range=(
                self.config.level - self.config.window/2,
                self.config.level + self.config.window/2
            )
        )
        
        overlay_hist, _ = np.histogram(
            self.overlay,
            bins=self.config.histogram_bins,
            range=(
                self.config.level - self.config.window/2,
                self.config.level + self.config.window/2
            )
        )
        
        return base_hist, overlay_hist
    
    def plot_histograms(self) -> Figure:
        """
        Plot intensity histograms.
        
        Returns:
            Matplotlib figure
        """
        base_hist, overlay_hist = self.calculate_histogram()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot histograms
        bin_edges = np.linspace(
            self.config.level - self.config.window/2,
            self.config.level + self.config.window/2,
            self.config.histogram_bins + 1
        )
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        ax.plot(bin_centers, base_hist, label="Base", alpha=0.7)
        ax.plot(bin_centers, overlay_hist, label="Overlay", alpha=0.7)
        
        ax.set_xlabel("Intensity")
        ax.set_ylabel("Frequency")
        ax.set_title("Intensity Histograms")
        ax.legend()
        
        return fig
    
    def create_controls(self) -> tk.Tk:
        """
        Create control window.
        
        Returns:
            Tkinter window
        """
        # Create window
        window = tk.Tk()
        window.title("Overlay Controls")
        
        # Mode selector
        tk.Label(window, text="Mode:").pack()
        mode_var = tk.StringVar(value=self.mode.name)
        tk.OptionMenu(
            window,
            mode_var,
            *[m.name for m in OverlayMode],
            command=lambda x: setattr(self, "mode", OverlayMode[x])
        ).pack()
        
        # Blend factor slider
        tk.Label(window, text="Blend Factor:").pack()
        blend_var = tk.DoubleVar(value=self.config.blend_factor)
        tk.Scale(
            window,
            from_=0.0,
            to=1.0,
            resolution=0.1,
            orient=tk.HORIZONTAL,
            variable=blend_var,
            command=lambda x: setattr(self.config, "blend_factor", float(x))
        ).pack()
        
        # Window/level controls
        tk.Label(window, text="Window:").pack()
        window_var = tk.IntVar(value=self.config.window)
        tk.Scale(
            window,
            from_=1,
            to=4000,
            orient=tk.HORIZONTAL,
            variable=window_var,
            command=lambda x: setattr(self.config, "window", int(x))
        ).pack()
        
        tk.Label(window, text="Level:").pack()
        level_var = tk.IntVar(value=self.config.level)
        tk.Scale(
            window,
            from_=-2000,
            to=2000,
            orient=tk.HORIZONTAL,
            variable=level_var,
            command=lambda x: setattr(self.config, "level", int(x))
        ).pack()
        
        return window