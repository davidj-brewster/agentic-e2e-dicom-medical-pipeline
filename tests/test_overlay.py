"""
Test suite for overlay visualization.
Tests visualization modes and analysis features.
"""
import numpy as np
import pytest
import tkinter as tk
from matplotlib.figure import Figure

from utils.overlay import (
    OverlayMode,
    OverlayVisualizer,
    VisualizationConfig
)


@pytest.fixture
def base_image() -> np.ndarray:
    """Fixture for base image"""
    return np.random.normal(100, 20, (64, 64))


@pytest.fixture
def overlay_image() -> np.ndarray:
    """Fixture for overlay image"""
    return np.random.normal(120, 25, (64, 64))


@pytest.fixture
def config() -> VisualizationConfig:
    """Fixture for visualization config"""
    return VisualizationConfig(
        window=1000,
        level=100,
        blend_factor=0.5,
        checkerboard_size=16,
        histogram_bins=50,
        split_position=0.5,
        colormap="gray",
        overlay_colormap="hot",
        overlay_alpha=0.5
    )


@pytest.fixture
def visualizer(
    base_image: np.ndarray,
    overlay_image: np.ndarray,
    config: VisualizationConfig
) -> OverlayVisualizer:
    """Fixture for overlay visualizer"""
    return OverlayVisualizer(base_image, overlay_image, config)


class TestOverlayVisualizer:
    """Tests for overlay visualizer"""
    
    def test_initialization(
        self,
        base_image: np.ndarray,
        overlay_image: np.ndarray,
        config: VisualizationConfig
    ):
        """Test visualizer initialization"""
        visualizer = OverlayVisualizer(base_image, overlay_image, config)
        assert visualizer.base is base_image
        assert visualizer.overlay is overlay_image
        assert visualizer.config == config
        assert visualizer.mode == OverlayMode.BLEND
        assert visualizer.difference_map is None
        assert not visualizer.metrics
    
    def test_initialization_validation(
        self,
        base_image: np.ndarray,
        config: VisualizationConfig
    ):
        """Test initialization validation"""
        # Test mismatched shapes
        invalid_overlay = np.random.normal(0, 1, (32, 32))
        with pytest.raises(ValueError):
            OverlayVisualizer(base_image, invalid_overlay, config)
    
    def test_normalization(self, visualizer: OverlayVisualizer):
        """Test image normalization"""
        # Test base image
        normalized = visualizer.normalize_image(visualizer.base)
        assert normalized.shape == visualizer.base.shape
        assert normalized.min() >= 0
        assert normalized.max() <= 1
        
        # Test overlay image
        normalized = visualizer.normalize_image(visualizer.overlay)
        assert normalized.shape == visualizer.overlay.shape
        assert normalized.min() >= 0
        assert normalized.max() <= 1
    
    def test_blend_mode(self, visualizer: OverlayVisualizer):
        """Test blend mode"""
        visualizer.mode = OverlayMode.BLEND
        result = visualizer.render()
        
        assert isinstance(result, np.ndarray)
        assert result.shape == visualizer.base.shape
        assert result.min() >= 0
        assert result.max() <= 1
    
    def test_checkerboard_mode(self, visualizer: OverlayVisualizer):
        """Test checkerboard mode"""
        visualizer.mode = OverlayMode.CHECKERBOARD
        result = visualizer.render()
        
        assert isinstance(result, np.ndarray)
        assert result.shape == visualizer.base.shape
        assert result.min() >= 0
        assert result.max() <= 1
    
    def test_difference_mode(self, visualizer: OverlayVisualizer):
        """Test difference mode"""
        visualizer.mode = OverlayMode.DIFFERENCE
        result = visualizer.render()
        
        assert isinstance(result, np.ndarray)
        assert result.shape == visualizer.base.shape
        assert result.min() >= 0
        assert result.max() <= 1
        assert visualizer.difference_map is not None
    
    def test_split_mode(self, visualizer: OverlayVisualizer):
        """Test split mode"""
        visualizer.mode = OverlayMode.SPLIT
        result = visualizer.render()
        
        assert isinstance(result, np.ndarray)
        assert result.shape == visualizer.base.shape
        assert result.min() >= 0
        assert result.max() <= 1
    
    def test_side_by_side_mode(self, visualizer: OverlayVisualizer):
        """Test side by side mode"""
        visualizer.mode = OverlayMode.SIDE_BY_SIDE
        result = visualizer.render()
        
        assert isinstance(result, Figure)
        assert len(result.axes) == 2
    
    def test_metrics(self, visualizer: OverlayVisualizer):
        """Test comparison metrics"""
        metrics = visualizer.calculate_metrics()
        
        assert "mse" in metrics
        assert "mae" in metrics
        assert "correlation" in metrics
        assert all(isinstance(v, float) for v in metrics.values())
        assert visualizer.difference_map is not None
    
    def test_histograms(self, visualizer: OverlayVisualizer):
        """Test histogram calculation"""
        # Test calculation
        base_hist, overlay_hist = visualizer.calculate_histogram()
        assert isinstance(base_hist, np.ndarray)
        assert isinstance(overlay_hist, np.ndarray)
        assert len(base_hist) == visualizer.config.histogram_bins
        assert len(overlay_hist) == visualizer.config.histogram_bins
        
        # Test plotting
        fig = visualizer.plot_histograms()
        assert isinstance(fig, Figure)
        assert len(fig.axes) == 1
    
    def test_controls(self, visualizer: OverlayVisualizer):
        """Test control window creation"""
        window = visualizer.create_controls()
        assert isinstance(window, tk.Tk)
        
        # Test mode change
        mode_menu = window.children["!optionmenu"]
        mode_menu.event_generate("<<ComboboxSelected>>")
        
        # Test slider updates
        for child in window.children.values():
            if isinstance(child, tk.Scale):
                child.event_generate("<ButtonRelease-1>")
        
        window.destroy()