"""
Test suite for CLI functionality.
Tests command parsing and execution.
"""
import asyncio
import json
from pathlib import Path
from typing import Dict, List
from unittest.mock import MagicMock, patch

import pytest
import tkinter as tk
from tkinter import ttk
from rich.console import Console

from utils.statistics import StatisticalTest
from utils.overlay import OverlayMode
from utils.registration import TransformType
from cli.commands import (
    CancelWorkflowCommand,
    ClearCacheCommand,
    Command,
    CompareSubjectsCommand,
    ExportReportCommand,
    ListCacheCommand,
    ListWorkflowsCommand,
    OptimizeCacheCommand,
    ProcessDatasetCommand,
    ProcessSubjectCommand,
    SetupEnvironmentCommand,
    ShowCacheCommand,
    ShowWorkflowCommand,
    ValidateCommand,
    View3DCommand,
    ViewResultsCommand
)
from cli.main import create_parser, main
from core.config import PipelineConfig
from tests.data.generate_test_data import create_test_subject
from tests.mocks import MockCommandRunner, create_mock_nifti


@pytest.fixture
def test_data_dir(tmp_path: Path) -> Path:
    """Fixture for test data directory"""
    return tmp_path / "test_data"


@pytest.fixture
def work_dir(tmp_path: Path) -> Path:
    """Fixture for working directory"""
    return tmp_path / "work"


@pytest.fixture
def test_subject(test_data_dir: Path) -> Dict[str, Path]:
    """Fixture for test subject data"""
    return create_test_subject(
        test_data_dir,
        "test_subject",
        shape=(64, 64, 64)
    )

@pytest.fixture
def test_subjects(test_data_dir: Path) -> List[Dict[str, Path]]:
    """Fixture for multiple test subjects"""
    subjects = []
    for i in range(3):
        subject = create_test_subject(
            test_data_dir / f"sub-{i:02d}",
            f"sub-{i:02d}",
            shape=(64, 64, 64)
        )
        subjects.append(subject)
    return subjects


@pytest.fixture
def mock_config(work_dir: Path) -> PipelineConfig:
    """Fixture for mock configuration"""
    config = PipelineConfig(
        working_dir=work_dir,
        output_dir=work_dir / "output",
        freesurfer_home=Path("/usr/local/freesurfer"),
        fsl_dir=Path("/usr/local/fsl"),
        anthropic_api_key="test-api-key"
    )
    return config


@pytest.fixture
def mock_command_runner() -> MockCommandRunner:
    """Fixture for mock command runner"""
    return MockCommandRunner()


class TestCLI:
    """Tests for CLI functionality"""
    
    def test_parser_creation(self):
        """Test argument parser creation"""
        parser = create_parser()
        
        # Verify global arguments
        args = parser.parse_args(["--log-level", "DEBUG"])
        assert args.log_level == "DEBUG"
        
        # Verify commands
        for name, command in {
            "setup-environment": SetupEnvironmentCommand,
            "validate": ValidateCommand,
            "process-subject": ProcessSubjectCommand,
            "process-dataset": ProcessDatasetCommand,
            "list-workflows": ListWorkflowsCommand,
            "show-workflow": ShowWorkflowCommand,
            "cancel-workflow": CancelWorkflowCommand,
            "list-cache": ListCacheCommand,
            "show-cache": ShowCacheCommand,
            "clear-cache": ClearCacheCommand,
            "optimize-cache": OptimizeCacheCommand,
            "view-results": ViewResultsCommand,
            "export-report": ExportReportCommand,
            "view-3d": View3DCommand,
            "compare-subjects": CompareSubjectsCommand
        }.items():
            args = parser.parse_args([name, "--help"])
            assert args.command == name
    
    @pytest.mark.asyncio
    async def test_view_results_command(
        self,
        test_subject: Dict[str, Path],
        mock_config: PipelineConfig
    ):
        """Test view results command"""
        # Test basic view
        args = ["view-results", "test_subject"]
        exit_code = await main(args)
        assert exit_code == 0
        
        # Test with view type
        args = [
            "view-results",
            "test_subject",
            "--view-type", "axial"
        ]
        exit_code = await main(args)
        assert exit_code == 0
        
        # Test with overlay
        args = [
            "view-results",
            "test_subject",
            "--overlay", "mask"
        ]
        exit_code = await main(args)
        assert exit_code == 0
        
        # Test with output
        output_path = mock_config.output_dir / "test.png"
        args = [
            "view-results",
            "test_subject",
            "--output", str(output_path)
        ]
        exit_code = await main(args)
        assert exit_code == 0
        assert output_path.exists()
    
    @pytest.mark.asyncio
    async def test_export_report_command(
        self,
        test_subject: Dict[str, Path],
        mock_config: PipelineConfig
    ):
        """Test export report command"""
        # Test HTML export
        args = ["export-report", "test_subject"]
        exit_code = await main(args)
        assert exit_code == 0
        
        # Test with output
        output_path = mock_config.output_dir / "report.html"
        args = [
            "export-report",
            "test_subject",
            "--output", str(output_path)
        ]
        exit_code = await main(args)
        assert exit_code == 0
        assert output_path.exists()
        
        # Test unsupported format
        args = [
            "export-report",
            "test_subject",
            "--format", "pdf"
        ]
        exit_code = await main(args)
        assert exit_code == 0  # Should succeed but log error
    
    @pytest.mark.asyncio
    async def test_view_3d_command(
        self,
        test_subject: Dict[str, Path],
        mock_config: PipelineConfig
    ):
        """Test view 3D command"""
        # Test basic view
        args = ["view-3d", "test_subject"]
        exit_code = await main(args)
        assert exit_code == 0
        
        # Test with surface mode
        args = [
            "view-3d",
            "test_subject",
            "--mode", "surface"
        ]
        exit_code = await main(args)
        assert exit_code == 0
        
        # Test with overlay
        args = [
            "view-3d",
            "test_subject",
            "--overlay", "mask"
        ]
        exit_code = await main(args)
        assert exit_code == 0
        
        # Test with measurement tool
        args = [
            "view-3d",
            "test_subject",
            "--tool", "distance"
        ]
        exit_code = await main(args)
        assert exit_code == 0
        
        # Test with output
        output_path = mock_config.output_dir / "test.png"
        args = [
            "view-3d",
            "test_subject",
            "--output", str(output_path)
        ]
        exit_code = await main(args)
        assert exit_code == 0
        assert output_path.exists()
        
        # Test measurement result persistence
        results_dir = mock_config.output_dir / "measurements"
        assert results_dir.exists()
        assert any(results_dir.glob("measurement_*.json"))
        assert (results_dir / "summary.json").exists()
    
    @pytest.mark.asyncio
    async def test_view_3d_error_handling(
        self,
        mock_config: PipelineConfig
    ):
        """Test view 3D error handling"""
        # Test invalid subject
        args = ["view-3d", "nonexistent"]
        exit_code = await main(args)
        assert exit_code == 1
        
        # Test invalid mode
        args = [
            "view-3d",
            "test_subject",
            "--mode", "invalid"
        ]
        exit_code = await main(args)
        assert exit_code == 1
        
        # Test invalid tool
        args = [
            "view-3d",
            "test_subject",
            "--tool", "invalid"
        ]
        exit_code = await main(args)
        assert exit_code == 1
        
        # Test invalid output path
        args = [
            "view-3d",
            "test_subject",
            "--output", "/invalid/path/test.png"
        ]
        exit_code = await main(args)
        assert exit_code == 1
    
    @pytest.mark.asyncio
    async def test_compare_subjects_basic(
        self,
        test_subjects: List[Dict[str, Path]],
        mock_config: PipelineConfig
    ):
        """Test basic comparison functionality"""
        # Test blend mode
        subject_ids = [s["id"] for s in test_subjects]
        args = ["compare-subjects"] + subject_ids
        exit_code = await main(args)
        assert exit_code == 0
        
        # Test different modes
        for mode in ["blend", "checkerboard", "difference", "split"]:
            args = [
                "compare-subjects",
                *subject_ids,
                "--mode", mode
            ]
            exit_code = await main(args)
            assert exit_code == 0
        
        # Test metrics
        for metric in ["mse", "correlation"]:
            args = [
                "compare-subjects",
                *subject_ids,
                "--metric", metric
            ]
            exit_code = await main(args)
            assert exit_code == 0
    
    @pytest.mark.asyncio
    async def test_compare_subjects_registration(
        self,
        test_subjects: List[Dict[str, Path]],
        mock_config: PipelineConfig
    ):
        """Test registration functionality"""
        subject_ids = [s["id"] for s in test_subjects]
        output_dir = mock_config.output_dir / "registration_test"
        
        # Test different transforms
        for transform in ["rigid", "affine", "bspline"]:
            args = [
                "compare-subjects",
                *subject_ids,
                "--registration", transform,
                "--output", str(output_dir)
            ]
            exit_code = await main(args)
            assert exit_code == 0
            
            # Verify transform files
            for subject_id in subject_ids[1:]:
                transform_path = output_dir / f"{subject_id}_transform.tfm"
                assert transform_path.exists()
    
    @pytest.mark.asyncio
    async def test_compare_subjects_visualization(
        self,
        test_subjects: List[Dict[str, Path]],
        mock_config: PipelineConfig
    ):
        """Test visualization functionality"""
        subject_ids = [s["id"] for s in test_subjects]
        output_dir = mock_config.output_dir / "visualization_test"
        
        # Test interactive mode
        args = [
            "compare-subjects",
            *subject_ids,
            "--interactive",
            "--output", str(output_dir)
        ]
        
        # Mock tkinter
        with patch("tkinter.Tk") as mock_tk:
            # Mock window
            mock_window = MagicMock()
            mock_tk.return_value = mock_window
            
            # Mock notebook
            mock_notebook = MagicMock()
            mock_window.children["!notebook"] = mock_notebook
            
            # Execute command
            exit_code = await main(args)
            assert exit_code == 0
            
            # Verify window creation
            mock_tk.assert_called_once()
            
            # Verify notebook creation
            assert len(mock_notebook.tabs()) == 2
            assert "Visualization" in mock_notebook.tab(0)["text"]
            assert "Statistics" in mock_notebook.tab(1)["text"]
            
            # Verify controls
            assert len(mock_window.children) > 0
            assert any(
                isinstance(child, ttk.Button)
                for child in mock_window.children.values()
            )
        
        # Test static mode with output
        args = [
            "compare-subjects",
            *subject_ids,
            "--output", str(output_dir)
        ]
        exit_code = await main(args)
        assert exit_code == 0
        
        # Verify outputs
        assert output_dir.exists()
        assert (output_dir / "statistics.json").exists()
        assert (output_dir / "distributions.png").exists()
        assert (output_dir / "correlation.png").exists()
        assert (output_dir / "report.html").exists()
        
        # Verify statistics
        with open(output_dir / "statistics.json") as f:
            stats = json.load(f)
        
        assert "roi_statistics" in stats
        assert "test_results" in stats
        for subject_id in subject_ids:
            assert subject_id in stats["roi_statistics"]
            subject_stats = stats["roi_statistics"][subject_id]
            assert "mean" in subject_stats
            assert "std" in subject_stats
            assert "volume" in subject_stats
            assert "skewness" in subject_stats
            assert "kurtosis" in subject_stats
            assert "iqr" in subject_stats
    
    @pytest.mark.asyncio
    async def test_compare_subjects_error_handling(
        self,
        mock_config: PipelineConfig
    ):
        """Test error handling"""
        # Test invalid subject
        args = ["compare-subjects", "nonexistent"]
        exit_code = await main(args)
        assert exit_code == 1
        
        # Test invalid mode
        args = [
            "compare-subjects",
            "test_subject",
            "--mode", "invalid"
        ]
        exit_code = await main(args)
        assert exit_code == 1
        
        # Test invalid registration
        args = [
            "compare-subjects",
            "test_subject",
            "--registration", "invalid"
        ]
        exit_code = await main(args)
        assert exit_code == 1
        
        # Test invalid output path
        args = [
            "compare-subjects",
            "test_subject",
            "--output", "/invalid/path"
        ]
        exit_code = await main(args)
        assert exit_code == 1
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling"""
        # Test invalid command
        args = ["invalid-command"]
        exit_code = await main(args)
        assert exit_code == 1
        
        # Test missing required argument
        args = ["view-results"]
        exit_code = await main(args)
        assert exit_code == 1
        
        # Test invalid subject
        args = ["view-results", "nonexistent"]
        exit_code = await main(args)
        assert exit_code == 1
    
    @pytest.mark.asyncio
    async def test_help_display(self):
        """Test help display"""
        # Test main help
        args = ["--help"]
        with pytest.raises(SystemExit) as exc_info:
            await main(args)
        assert exc_info.value.code == 0
        
        # Test command help
        for command in [
            "setup-environment",
            "validate",
            "process-subject",
            "process-dataset",
            "list-workflows",
            "show-workflow",
            "cancel-workflow",
            "list-cache",
            "show-cache",
            "clear-cache",
            "optimize-cache",
            "view-results",
            "export-report",
            "view-3d",
            "compare-subjects"
        ]:
            args = [command, "--help"]
            with pytest.raises(SystemExit) as exc_info:
                await main(args)
            assert exc_info.value.code == 0