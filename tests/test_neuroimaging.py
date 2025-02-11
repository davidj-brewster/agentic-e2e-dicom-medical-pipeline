"""
Test suite for neuroimaging-specific functionality.
Tests FSL/FreeSurfer integration, image processing, and analysis.
"""
import asyncio
from pathlib import Path
from typing import Dict, List, Tuple
from uuid import UUID

import nibabel as nib
import numpy as np
import pytest
from scipy.ndimage import label

from agents.analyzer import AnalyzerAgent
from agents.base import AgentConfig
from agents.coordinator import CoordinatorAgent
from agents.preprocessor import PreprocessingAgent
from agents.visualizer import VisualizerAgent
from core.messages import MessageQueue, create_command
from core.workflow import ResourceRequirements, WorkflowManager
from tests.data.generate_test_data import (
    create_synthetic_brain,
    create_test_subject
)
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
def command_runner() -> MockCommandRunner:
    """Fixture for mock command runner"""
    return setup_mock_command_runner()


@pytest.fixture
def message_queue() -> MockMessageQueue:
    """Fixture for message queue"""
    return MockMessageQueue()


@pytest.fixture
def workflow_manager() -> WorkflowManager:
    """Fixture for workflow manager"""
    return WorkflowManager()


@pytest.fixture
def coordinator_config(work_dir: Path) -> AgentConfig:
    """Fixture for coordinator configuration"""
    return AgentConfig(
        name="coordinator",
        capabilities={"workflow_management", "agent_coordination"},
        resource_limits=ResourceRequirements(
            cpu_cores=2,
            memory_gb=4.0
        ),
        working_dir=work_dir / "coordinator"
    )


class TestImageProcessing:
    """Tests for image processing functionality"""
    
    def test_synthetic_brain_generation(self):
        """Test synthetic brain volume generation"""
        volume = create_synthetic_brain(shape=(64, 64, 64))
        
        assert volume.shape == (64, 64, 64)
        assert np.any(volume > 0)  # Has brain tissue
        assert np.any(volume < 0.2)  # Has CSF-like values
        assert np.any((volume > 0.6) & (volume < 0.9))  # Has GM-like values
    
    @pytest.mark.asyncio
    async def test_preprocessing(
        self,
        test_subject: Dict[str, Path],
        work_dir: Path,
        command_runner: MockCommandRunner,
        message_queue: MessageQueue
    ):
        """Test preprocessing pipeline"""
        # Initialize preprocessor agent
        preprocessor = PreprocessingAgent(
            coordinator_id="test_coordinator",
            message_queue=message_queue
        )
        
        # Run preprocessing
        await preprocessor.run_preprocessing(
            subject_id="test_subject",
            input_files=test_subject
        )
        
        # Verify command executions
        executions = command_runner.get_executions()
        commands = [e.command for e in executions]
        
        assert "fslmaths" in commands  # Normalization
        assert "flirt" in commands  # Registration
        
        # Verify outputs exist
        assert (work_dir / "test_subject" / "prep" / "T1_norm.nii.gz").exists()
        assert (work_dir / "test_subject" / "reg" / "T2_FLAIR_reg.nii.gz").exists()
    
    @pytest.mark.asyncio
    async def test_registration(
        self,
        test_subject: Dict[str, Path],
        command_runner: MockCommandRunner
    ):
        """Test image registration"""
        # Create test files
        moving_path = test_subject["T2_FLAIR"]
        fixed_path = test_subject["T1"]
        output_path = Path("registered.nii.gz")
        
        # Run mock registration
        success, output, error = await command_runner.execute(
            "flirt",
            in_=str(moving_path),
            ref=str(fixed_path),
            out=str(output_path)
        )
        
        assert success
        assert output_path.exists()
        
        # Verify output dimensions match reference
        fixed_img = nib.load(fixed_path)
        output_img = nib.load(output_path)
        assert output_img.shape == fixed_img.shape


class TestSegmentationAnalysis:
    """Tests for segmentation and analysis functionality"""
    
    @pytest.mark.asyncio
    async def test_segmentation(
        self,
        test_subject: Dict[str, Path],
        work_dir: Path,
        command_runner: MockCommandRunner,
        message_queue: MessageQueue
    ):
        """Test segmentation pipeline"""
        # Initialize analyzer agent
        analyzer = AnalyzerAgent(
            coordinator_id="test_coordinator",
            message_queue=message_queue
        )
        
        # Run analysis
        await analyzer.run_analysis(
            subject_id="test_subject",
            t1_path=test_subject["T1"],
            flair_path=test_subject["T2_FLAIR"],
            working_dir=work_dir
        )
        
        # Verify FIRST execution
        first_executions = command_runner.get_executions("run_first_all")
        assert len(first_executions) > 0
        
        # Verify segmentation outputs
        seg_dir = work_dir / "test_subject" / "seg"
        assert any(seg_dir.glob("*.nii.gz"))
    
    def test_clustering_analysis(self):
        """Test intensity clustering analysis"""
        # Create test data with artificial clusters
        data = np.zeros((64, 64, 64))
        
        # Add some clusters with different intensities
        centers = [(20, 20, 20), (40, 40, 40)]
        radii = [5, 3]
        intensities = [1.0, 2.0]
        
        for center, radius, intensity in zip(centers, radii, intensities):
            x, y, z = np.ogrid[
                :64,
                :64,
                :64
            ]
            mask = (
                (x - center[0])**2 +
                (y - center[1])**2 +
                (z - center[2])**2
            ) <= radius**2
            data[mask] = intensity
        
        # Add noise
        data += np.random.normal(0, 0.1, data.shape)
        
        # Create mask
        mask = data > 0
        
        # Perform clustering
        labeled_data, num_clusters = label(mask)
        assert num_clusters == len(centers)
        
        # Verify cluster properties
        for i in range(1, num_clusters + 1):
            cluster_mask = labeled_data == i
            cluster_intensities = data[cluster_mask]
            
            assert len(cluster_intensities) > 0
            assert np.mean(cluster_intensities) > 0


class TestVisualization:
    """Tests for visualization functionality"""
    
    @pytest.mark.asyncio
    async def test_multiplanar_visualization(
        self,
        test_subject: Dict[str, Path],
        work_dir: Path,
        message_queue: MessageQueue
    ):
        """Test multi-planar visualization generation"""
        # Initialize visualizer agent
        visualizer = VisualizerAgent(
            coordinator_id="test_coordinator",
            message_queue=message_queue
        )
        
        # Create test mask and anomalies
        mask_path, _ = create_mock_nifti((64, 64, 64))
        anomaly_metrics = [
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
        
        # Generate visualization
        output_path = await visualizer.generate_multiplanar_views(
            t1_path=test_subject["T1"],
            flair_path=test_subject["T2_FLAIR"],
            mask_path=mask_path,
            anomaly_clusters=anomaly_metrics
        )
        
        assert output_path is not None
        assert output_path.exists()
    
    @pytest.mark.asyncio
    async def test_report_generation(
        self,
        test_subject: Dict[str, Path],
        work_dir: Path,
        message_queue: MessageQueue
    ):
        """Test analysis report generation"""
        # Initialize visualizer agent
        visualizer = VisualizerAgent(
            coordinator_id="test_coordinator",
            message_queue=message_queue
        )
        
        # Create test results
        analysis_results = {
            "L_Thal": {
                "segmentation": {
                    "region_name": "L_Thal",
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
        
        visualization_paths = {
            "L_Thal_mpr": work_dir / "vis" / "mpr" / "L_Thal.png"
        }
        
        # Generate report
        report_path = await visualizer.generate_report(
            analysis_results,
            visualization_paths
        )
        
        assert report_path is not None
        assert report_path.exists()
        
        # Verify report content
        with open(report_path) as f:
            content = f.read()
            assert "L_Thal" in content
            assert "Volume" in content
            assert "Anomalous Clusters" in content


class TestEndToEnd:
    """End-to-end tests for complete pipeline"""
    
    @pytest.mark.asyncio
    async def test_complete_pipeline(
        self,
        test_subject: Dict[str, Path],
        work_dir: Path,
        command_runner: MockCommandRunner,
        message_queue: MockMessageQueue,
        workflow_manager: WorkflowManager,
        coordinator_config: AgentConfig
    ):
        """Test complete processing pipeline"""
        # Initialize coordinator
        coordinator = CoordinatorAgent(
            config=coordinator_config,
            workflow_manager=workflow_manager
        )
        
        # Initialize specialized agents
        preprocessor = PreprocessingAgent(
            coordinator_id=coordinator_config.name,
            message_queue=message_queue
        )
        
        analyzer = AnalyzerAgent(
            coordinator_id=coordinator_config.name,
            message_queue=message_queue
        )
        
        visualizer = VisualizerAgent(
            coordinator_id=coordinator_config.name,
            message_queue=message_queue
        )
        
        # Register agents with coordinator
        await coordinator._register_agent(
            "preprocessor",
            ["preprocessing", "registration"]
        )
        await coordinator._register_agent(
            "analyzer",
            ["segmentation", "analysis"]
        )
        await coordinator._register_agent(
            "visualizer",
            ["visualization", "reporting"]
        )
        
        # Initialize workflow
        workflow_id = await coordinator._initialize_workflow(
            subject_id="test_subject",
            input_data={
                "t1": str(test_subject["T1"]),
                "t2_flair": str(test_subject["T2_FLAIR"])
            }
        )
        
        # Run preprocessing
        await preprocessor.run_preprocessing(
            subject_id="test_subject",
            input_files=test_subject
        )
        
        # Get preprocessed files
        preprocessed_files = {
            "T1": work_dir / "test_subject" / "prep" / "T1_norm.nii.gz",
            "T2_FLAIR": work_dir / "test_subject" / "reg" / "T2_FLAIR_reg.nii.gz"
        }
        
        # Run analysis
        await analyzer.run_analysis(
            subject_id="test_subject",
            t1_path=preprocessed_files["T1"],
            flair_path=preprocessed_files["T2_FLAIR"],
            working_dir=work_dir
        )
        
        # Get analysis results
        analysis_results = None
        for message in message_queue.get_messages(coordinator_config.name):
            if "analysis_results" in message.payload:
                analysis_results = message.payload["analysis_results"]
                break
        
        assert analysis_results is not None
        
        # Run visualization
        await visualizer.run_visualization(
            subject_id="test_subject",
            t1_path=preprocessed_files["T1"],
            flair_path=preprocessed_files["T2_FLAIR"],
            analysis_results=analysis_results,
            working_dir=work_dir
        )
        
        # Verify workflow completion
        status = await coordinator._get_workflow_status(workflow_id)
        assert status["status"] == "completed"
        
        # Verify outputs
        assert (work_dir / "test_subject" / "vis" / "reports").exists()
        assert any((work_dir / "test_subject" / "vis" / "mpr").glob("*.png"))
        
        # Verify workflow caching
        cached = await workflow_manager.get_cached_workflow("test_subject")
        assert cached is not None
        assert cached.workflow_id == workflow_id