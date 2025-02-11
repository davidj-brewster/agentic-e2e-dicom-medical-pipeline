"""
Test suite for preprocessing agent functionality.
Tests FSL integration, image preprocessing, and quality control.
"""
import asyncio
from pathlib import Path
from typing import Dict

import nibabel as nib
import numpy as np
import pytest

from agents.base import AgentConfig
from agents.preprocessor import (
    PreprocessingAgent,
    PreprocessingConfig,
    PreprocessingMetrics
)
from core.config import FSLConfig, FreeSurferConfig, ProcessingConfig
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
def preprocessor_config(work_dir: Path) -> AgentConfig:
    """Fixture for preprocessor agent configuration"""
    return AgentConfig(
        name="preprocessor",
        capabilities={"preprocessing", "registration"},
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
def preprocessing_config(work_dir: Path) -> PreprocessingConfig:
    """Fixture for preprocessing configuration"""
    return PreprocessingConfig(
        fsl=FSLConfig(
            fsl_dir=Path("/usr/local/fsl"),
            standard_brain=Path("/usr/local/fsl/data/standard/MNI152_T1_2mm_brain.nii.gz"),
            registration_cost="corratio",
            interpolation="trilinear"
        ),
        freesurfer=FreeSurferConfig(
            freesurfer_home=Path("/usr/local/freesurfer"),
            subjects_dir=work_dir / "subjects",
            license_file=Path("/usr/local/freesurfer/license.txt")
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
def preprocessor_agent(
    preprocessor_config: AgentConfig,
    preprocessing_config: PreprocessingConfig,
    message_queue: MockMessageQueue
) -> PreprocessingAgent:
    """Fixture for preprocessor agent"""
    return PreprocessingAgent(
        config=preprocessor_config,
        preprocessing_config=preprocessing_config,
        message_queue=message_queue
    )


class TestPreprocessorAgent:
    """Tests for preprocessor agent functionality"""
    
    @pytest.mark.asyncio
    async def test_initialization(self, preprocessor_agent: PreprocessingAgent):
        """Test agent initialization"""
        await preprocessor_agent._initialize()
        assert preprocessor_agent.env_setup_verified
        assert preprocessor_agent.state.status == "ready"
    
    @pytest.mark.asyncio
    async def test_input_validation(
        self,
        preprocessor_agent: PreprocessingAgent,
        test_subject: Dict[str, Path]
    ):
        """Test input file validation"""
        # Initialize agent
        await preprocessor_agent._initialize()
        
        # Validate inputs
        metadata = await preprocessor_agent._validate_input(test_subject)
        
        # Check metadata
        assert "T1" in metadata
        assert "T2_FLAIR" in metadata
        assert metadata["T1"].dimensions == (64, 64, 64)
        assert len(metadata["T1"].voxel_size) == 3
    
    @pytest.mark.asyncio
    async def test_preprocessing_pipeline(
        self,
        preprocessor_agent: PreprocessingAgent,
        test_subject: Dict[str, Path],
        message_queue: MockMessageQueue
    ):
        """Test complete preprocessing pipeline"""
        # Initialize agent
        await preprocessor_agent._initialize()
        
        # Run preprocessing
        await preprocessor_agent._handle_command(
            Message(
                sender="coordinator",
                recipient="preprocessor",
                message_type=MessageType.COMMAND,
                payload={
                    "command": "preprocess_subject",
                    "parameters": {
                        "subject_id": "test_subject",
                        "input_files": {
                            k: str(v) for k, v in test_subject.items()
                        }
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
        assert len(status_updates) >= 3  # At least initial, middle, and completion
        assert status_updates[-1].payload.progress == 1.0
        assert status_updates[-1].payload.state == "completed"
        
        # Verify results
        assert len(data_messages) == 1
        results = data_messages[0].payload
        assert results.data_type == "preprocessing_results"
        assert "T1" in results.content["preprocessed_files"]
        assert "T2_FLAIR" in results.content["preprocessed_files"]
        assert "metrics" in results.content
    
    @pytest.mark.asyncio
    async def test_bias_correction(
        self,
        preprocessor_agent: PreprocessingAgent,
        test_subject: Dict[str, Path]
    ):
        """Test bias field correction"""
        # Initialize agent
        await preprocessor_agent._initialize()
        
        # Apply bias correction
        output_path = preprocessor_agent.config.working_dir / "bias_corr.nii.gz"
        success, error = await preprocessor_agent._apply_bias_correction(
            test_subject["T1"],
            output_path
        )
        
        assert success
        assert error is None
        assert output_path.exists()
    
    @pytest.mark.asyncio
    async def test_error_handling(
        self,
        preprocessor_agent: PreprocessingAgent,
        message_queue: MockMessageQueue
    ):
        """Test error handling"""
        # Initialize agent
        await preprocessor_agent._initialize()
        
        # Try to process non-existent files
        await preprocessor_agent._handle_command(
            Message(
                sender="coordinator",
                recipient="preprocessor",
                message_type=MessageType.COMMAND,
                payload={
                    "command": "preprocess_subject",
                    "parameters": {
                        "subject_id": "test_subject",
                        "input_files": {
                            "T1": "nonexistent.nii.gz",
                            "T2_FLAIR": "nonexistent.nii.gz"
                        }
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
        assert "not found" in error_messages[0].payload.message
    
    @pytest.mark.asyncio
    async def test_resource_management(
        self,
        preprocessor_agent: PreprocessingAgent,
        test_subject: Dict[str, Path]
    ):
        """Test resource management"""
        # Initialize agent
        await preprocessor_agent._initialize()
        
        # Check initial resource state
        assert preprocessor_agent.state.resource_usage.get("cpu", 0) == 0
        assert preprocessor_agent.state.resource_usage.get("memory", 0) == 0
        
        # Start preprocessing
        await preprocessor_agent._handle_command(
            Message(
                sender="coordinator",
                recipient="preprocessor",
                message_type=MessageType.COMMAND,
                payload={
                    "command": "preprocess_subject",
                    "parameters": {
                        "subject_id": "test_subject",
                        "input_files": {
                            k: str(v) for k, v in test_subject.items()
                        }
                    }
                }
            )
        )
        
        # Verify resources were allocated and released
        assert preprocessor_agent.state.resource_usage.get("cpu", 0) == 0
        assert preprocessor_agent.state.resource_usage.get("memory", 0) == 0
    
    @pytest.mark.asyncio
    async def test_metrics_calculation(
        self,
        preprocessor_agent: PreprocessingAgent,
        test_subject: Dict[str, Path]
    ):
        """Test preprocessing metrics calculation"""
        # Initialize agent
        await preprocessor_agent._initialize()
        
        # Run preprocessing
        await preprocessor_agent._handle_command(
            Message(
                sender="coordinator",
                recipient="preprocessor",
                message_type=MessageType.COMMAND,
                payload={
                    "command": "preprocess_subject",
                    "parameters": {
                        "subject_id": "test_subject",
                        "input_files": {
                            k: str(v) for k, v in test_subject.items()
                        }
                    }
                }
            )
        )
        
        # Get metrics from results
        data_messages = [
            m for m in preprocessor_agent.message_queue.get_message_history()
            if m.message_type == MessageType.DATA
        ]
        
        metrics = PreprocessingMetrics(**data_messages[0].payload.content["metrics"])
        
        # Verify metrics
        assert metrics.snr > 0
        assert metrics.contrast_to_noise > 0
        assert metrics.registration_cost is not None