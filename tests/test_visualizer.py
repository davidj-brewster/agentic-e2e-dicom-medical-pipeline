"""
Test suite for visualizer agent functionality.
Tests visualization generation and report creation.
"""
import asyncio
from pathlib import Path
from typing import Dict

import nibabel as nib
import numpy as np
import pytest

from agents.base import AgentConfig
from agents.visualizer import VisualizerAgent
from core.config import VisualizationConfig
from core.messages import Message, MessageType
from core.workflow import ResourceRequirements
from tests.data.generate_test_data import create_test_subject
from tests.mocks import (
    MockCommandRunner,
    MockMessageQueue,
    create_mock_nifti
)


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
def visualizer_config(work_dir: Path) -> AgentConfig:
    """Fixture for visualizer agent configuration"""
    return AgentConfig(
        name="visualizer",
        capabilities={"visualization", "reporting"},
        resource_limits=ResourceRequirements(
            cpu_cores=2,
            memory_gb=4.0
        ),
        working_dir=work_dir,
        environment={}
    )


@pytest.fixture
def visualization_config() -> VisualizationConfig:
    """Fixture for visualization configuration"""
    return VisualizationConfig(
        output_dpi=300,
        slice_spacing=5,
        overlay_alpha=0.35,
        anomaly_cmap="hot",
        background_cmap="gray"
    )


@pytest.fixture
def message_queue() -> MockMessageQueue:
    """Fixture for mock message queue"""
    return MockMessageQueue()


@pytest.fixture
def visualizer_agent(
    visualizer_config: AgentConfig,
    visualization_config: VisualizationConfig,
    message_queue: MockMessageQueue
) -> VisualizerAgent:
    """Fixture for visualizer agent"""
    return VisualizerAgent(
        config=visualizer_config,
        visualization_config=visualization_config,
        message_queue=message_queue
    )


@pytest.fixture
def mock_analysis_results(test_subject: Dict[str, Path]) -> Dict[str, Any]:
    """Fixture for mock analysis results"""
    # Create mock segmentation mask
    mask_path, _ = create_mock_nifti((64, 64, 64))
    
    return {
        "L_Thal": {
            "segmentation": {
                "region_name": "L_Thal",
                "mask_path": str(mask_path),
                "volume": 1000.0,
                "center_of_mass": (32, 32, 32),
                "voxel_count": 1000
            },
            "clusters": [
                {
                    "cluster_id": 1,
                    "size": 100,
                    "mean_intensity": 1.5,
                    "std_intensity": 0.2,
                    "center": (32, 32, 32),
                    "bounding_box": ((30, 30, 30), (34, 34, 34)),
                    "outlier_score": 2.0
                }
            ],
            "anomalies": [
                {
                    "cluster_id": 1,
                    "size": 100,
                    "mean_intensity": 1.5,
                    "std_intensity": 0.2,
                    "center": (32, 32, 32),
                    "bounding_box": ((30, 30, 30), (34, 34, 34)),
                    "outlier_score": 2.0
                }
            ]
        }
    }


class TestVisualizerAgent:
    """Tests for visualizer agent functionality"""
    
    @pytest.mark.asyncio
    async def test_initialization(self, visualizer_agent: VisualizerAgent):
        """Test agent initialization"""
        await visualizer_agent._initialize()
        assert visualizer_agent.state.status == "ready"
    
    @pytest.mark.asyncio
    async def test_visualization_pipeline(
        self,
        visualizer_agent: VisualizerAgent,
        test_subject: Dict[str, Path],
        mock_analysis_results: Dict[str, Any],
        message_queue: MockMessageQueue
    ):
        """Test complete visualization pipeline"""
        # Initialize agent
        await visualizer_agent._initialize()
        
        # Run visualization
        await visualizer_agent._handle_command(
            Message(
                sender="coordinator",
                recipient="visualizer",
                message_type=MessageType.COMMAND,
                payload={
                    "command": "visualize_subject",
                    "parameters": {
                        "subject_id": "test_subject",
                        "t1_path": str(test_subject["T1"]),
                        "flair_path": str(test_subject["T2_FLAIR"]),
                        "analysis_results": mock_analysis_results
                    }
                }
            )
        )
        
        # Check messages
        messages = message_queue.get_message_history()
        status_updates = [
            m for m in messages
            if m.message_type == MessageType.STATUS_UPDATE
        ]
        data_messages = [
            m for m in messages
            if m.message_type == MessageType.DATA
        ]
        
        # Verify progress updates
        assert len(status_updates) >= 3  # Initial, progress, completion
        assert status_updates[-1].payload.progress == 1.0
        assert status_updates[-1].payload.state == "completed"
        
        # Verify results
        assert len(data_messages) == 1
        results = data_messages[0].payload
        assert results.data_type == "visualization_results"
        assert "visualization_paths" in results.content
        assert "report_path" in results.content
        
        # Check output files
        vis_paths = results.content["visualization_paths"]
        for path_str in vis_paths.values():
            assert Path(path_str).exists()
        
        report_path = Path(results.content["report_path"])
        assert report_path.exists()
    
    @pytest.mark.asyncio
    async def test_error_handling(
        self,
        visualizer_agent: VisualizerAgent,
        message_queue: MockMessageQueue
    ):
        """Test error handling"""
        # Initialize agent
        await visualizer_agent._initialize()
        
        # Try to visualize non-existent files
        await visualizer_agent._handle_command(
            Message(
                sender="coordinator",
                recipient="visualizer",
                message_type=MessageType.COMMAND,
                payload={
                    "command": "visualize_subject",
                    "parameters": {
                        "subject_id": "test_subject",
                        "t1_path": "nonexistent.nii.gz",
                        "flair_path": "nonexistent.nii.gz",
                        "analysis_results": {}
                    }
                }
            )
        )
        
        # Check error messages
        error_messages = [
            m for m in message_queue.get_message_history()
            if m.message_type == MessageType.ERROR
        ]
        
        assert len(error_messages) > 0
        assert "failed" in error_messages[0].payload.message
    
    @pytest.mark.asyncio
    async def test_report_generation(
        self,
        visualizer_agent: VisualizerAgent,
        mock_analysis_results: Dict[str, Any]
    ):
        """Test report generation"""
        # Initialize agent
        await visualizer_agent._initialize()
        
        # Create mock visualization paths
        vis_dir = visualizer_agent.config.working_dir / "test_subject" / "vis"
        vis_dir.mkdir(parents=True)
        
        mpr_path = vis_dir / "mpr" / "test.png"
        mpr_path.parent.mkdir()
        mpr_path.touch()
        
        visualization_paths = {
            "test_mpr": mpr_path
        }
        
        # Generate report
        report_path = await visualizer_agent._generate_report(
            mock_analysis_results,
            visualization_paths
        )
        
        # Verify report
        assert report_path is not None
        assert report_path.exists()
        
        # Check report content
        with open(report_path) as f:
            content = f.read()
            assert "Analysis Report" in content
            assert "L_Thal" in content
            assert "1000.0" in content  # Volume
            assert "2.0" in content  # Max outlier score
            assert str(mpr_path.relative_to(visualizer_agent.subject_dir)) in content
    
    @pytest.mark.asyncio
    async def test_resource_management(
        self,
        visualizer_agent: VisualizerAgent,
        test_subject: Dict[str, Path],
        mock_analysis_results: Dict[str, Any]
    ):
        """Test resource management"""
        # Initialize agent
        await visualizer_agent._initialize()
        
        # Check initial resource state
        assert visualizer_agent.state.resource_usage.get("cpu", 0) == 0
        assert visualizer_agent.state.resource_usage.get("memory", 0) == 0
        
        # Start visualization
        await visualizer_agent._handle_command(
            Message(
                sender="coordinator",
                recipient="visualizer",
                message_type=MessageType.COMMAND,
                payload={
                    "command": "visualize_subject",
                    "parameters": {
                        "subject_id": "test_subject",
                        "t1_path": str(test_subject["T1"]),
                        "flair_path": str(test_subject["T2_FLAIR"]),
                        "analysis_results": mock_analysis_results
                    }
                }
            )
        )
        
        # Verify resources were allocated and released
        assert visualizer_agent.state.resource_usage.get("cpu", 0) == 0
        assert visualizer_agent.state.resource_usage.get("memory", 0) == 0
