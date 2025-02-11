"""
Test suite for analyzer agent functionality.
Tests segmentation, clustering, and anomaly detection.
"""
import asyncio
from pathlib import Path
from typing import Dict

import nibabel as nib
import numpy as np
import pytest

from agents.base import AgentConfig
from agents.analyzer import (
    AnalyzerAgent,
    AnalyzerConfig
)
from core.config import (
    ClusteringConfig,
    ProcessingConfig,
    SegmentationConfig
)
from core.messages import Message, MessageType
from core.workflow import ResourceRequirements
from tests.data.generate_test_data import create_test_subject
from tests.mocks import (
    MockCommandRunner,
    MockMessageQueue,
    create_mock_nifti,
    setup_mock_command_runner
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
def analyzer_config(work_dir: Path) -> AgentConfig:
    """Fixture for analyzer agent configuration"""
    return AgentConfig(
        name="analyzer",
        capabilities={"segmentation", "analysis"},
        resource_limits=ResourceRequirements(
            cpu_cores=4,
            memory_gb=8.0
        ),
        working_dir=work_dir,
        environment={
            "FSLDIR": "/usr/local/fsl",
            "FREESURFER_HOME": "/usr/local/freesurfer"
        }
    )


@pytest.fixture
def analysis_config() -> AnalyzerConfig:
    """Fixture for analysis configuration"""
    return AnalyzerConfig(
        segmentation=SegmentationConfig(
            regions_of_interest={
                "L_Thal": 10,
                "R_Thal": 49,
                "L_Caud": 11,
                "R_Caud": 50
            },
            probability_threshold=0.5,
            min_region_size=10
        ),
        clustering=ClusteringConfig(
            outlier_threshold=2.0,
            min_cluster_size=5,
            connectivity=26
        ),
        processing=ProcessingConfig(
            normalize_intensity=True,
            bias_correction=True,
            skull_strip=True,
            registration_dof=12,
            smoothing_fwhm=2.0
        )
    )


@pytest.fixture
def message_queue() -> MockMessageQueue:
    """Fixture for mock message queue"""
    return MockMessageQueue()


@pytest.fixture
def command_runner() -> MockCommandRunner:
    """Fixture for mock command runner"""
    return setup_mock_command_runner()


@pytest.fixture
def analyzer_agent(
    analyzer_config: AgentConfig,
    analysis_config: AnalyzerConfig,
    message_queue: MockMessageQueue
) -> AnalyzerAgent:
    """Fixture for analyzer agent"""
    return AnalyzerAgent(
        config=analyzer_config,
        analyzer_config=analysis_config,
        message_queue=message_queue
    )


class TestAnalyzerAgent:
    """Tests for analyzer agent functionality"""
    
    @pytest.mark.asyncio
    async def test_initialization(self, analyzer_agent: AnalyzerAgent):
        """Test agent initialization"""
        await analyzer_agent._initialize()
        assert analyzer_agent.env_setup_verified
        assert analyzer_agent.state.status == "ready"
    
    @pytest.mark.asyncio
    async def test_segmentation_pipeline(
        self,
        analyzer_agent: AnalyzerAgent,
        test_subject: Dict[str, Path],
        message_queue: MockMessageQueue
    ):
        """Test complete segmentation pipeline"""
        # Initialize agent
        await analyzer_agent._initialize()
        
        # Run analysis
        await analyzer_agent._handle_command(
            Message(
                sender="coordinator",
                recipient="analyzer",
                message_type=MessageType.COMMAND,
                payload={
                    "command": "analyze_subject",
                    "parameters": {
                        "subject_id": "test_subject",
                        "t1_path": str(test_subject["T1"]),
                        "flair_path": str(test_subject["T2_FLAIR"])
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
        assert len(status_updates) >= 4  # Initial, segmentation, masks, completion
        assert status_updates[-1].payload.progress == 1.0
        assert status_updates[-1].payload.state == "completed"
        
        # Verify results
        assert len(data_messages) == 1
        results = data_messages[0].payload
        assert results.data_type == "analysis_results"
        assert "analysis_results" in results.content
        
        # Check result structure
        analysis_results = results.content["analysis_results"]
        for region in analyzer_agent.analyzer_config.segmentation.regions_of_interest:
            assert region in analysis_results
            region_results = analysis_results[region]
            assert "segmentation" in region_results
            assert "clusters" in region_results
            assert "anomalies" in region_results
    
    @pytest.mark.asyncio
    async def test_error_handling(
        self,
        analyzer_agent: AnalyzerAgent,
        message_queue: MockMessageQueue
    ):
        """Test error handling"""
        # Initialize agent
        await analyzer_agent._initialize()
        
        # Try to analyze non-existent files
        await analyzer_agent._handle_command(
            Message(
                sender="coordinator",
                recipient="analyzer",
                message_type=MessageType.COMMAND,
                payload={
                    "command": "analyze_subject",
                    "parameters": {
                        "subject_id": "test_subject",
                        "t1_path": "nonexistent.nii.gz",
                        "flair_path": "nonexistent.nii.gz"
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
    async def test_resource_management(
        self,
        analyzer_agent: AnalyzerAgent,
        test_subject: Dict[str, Path]
    ):
        """Test resource management"""
        # Initialize agent
        await analyzer_agent._initialize()
        
        # Check initial resource state
        assert analyzer_agent.state.resource_usage.get("cpu", 0) == 0
        assert analyzer_agent.state.resource_usage.get("memory", 0) == 0
        
        # Start analysis
        await analyzer_agent._handle_command(
            Message(
                sender="coordinator",
                recipient="analyzer",
                message_type=MessageType.COMMAND,
                payload={
                    "command": "analyze_subject",
                    "parameters": {
                        "subject_id": "test_subject",
                        "t1_path": str(test_subject["T1"]),
                        "flair_path": str(test_subject["T2_FLAIR"])
                    }
                }
            )
        )
        
        # Verify resources were allocated and released
        assert analyzer_agent.state.resource_usage.get("cpu", 0) == 0
        assert analyzer_agent.state.resource_usage.get("memory", 0) == 0
    
    @pytest.mark.asyncio
    async def test_validation(
        self,
        analyzer_agent: AnalyzerAgent,
        test_subject: Dict[str, Path]
    ):
        """Test segmentation validation"""
        # Initialize agent
        await analyzer_agent._initialize()
        
        # Create invalid segmentation (empty)
        seg_path = analyzer_agent.config.working_dir / "invalid_seg.nii.gz"
        empty_data = np.zeros((64, 64, 64))
        nib.save(nib.Nifti1Image(empty_data, np.eye(4)), str(seg_path))
        
        # Run analysis with invalid segmentation
        await analyzer_agent._handle_command(
            Message(
                sender="coordinator",
                recipient="analyzer",
                message_type=MessageType.COMMAND,
                payload={
                    "command": "analyze_subject",
                    "parameters": {
                        "subject_id": "test_subject",
                        "t1_path": str(test_subject["T1"]),
                        "flair_path": str(test_subject["T2_FLAIR"])
                    }
                }
            )
        )
        
        # Check error messages
        error_messages = [
            m for m in analyzer_agent.message_queue.get_message_history()
            if m.message_type == MessageType.ERROR
        ]
        
        assert len(error_messages) > 0
        assert any("validation failed" in m.payload.message for m in error_messages)