"""
Test suite for pipeline orchestration.
Tests workflow execution, caching, and resource management.
"""
import asyncio
from pathlib import Path
from typing import Dict

import pytest

from core.config import PipelineConfig, load_config
from core.pipeline import Pipeline
from tests.data.generate_test_data import create_test_subject
from tests.mocks import (
    MockCommandRunner,
    MockMessageQueue,
    create_mock_nifti
)
from utils.pipeline import ResourceUsage, WorkflowPattern


@pytest.fixture
def test_data_dir(tmp_path: Path) -> Path:
    """Fixture for test data directory"""
    return tmp_path / "test_data"


@pytest.fixture
def work_dir(tmp_path: Path) -> Path:
    """Fixture for working directory"""
    return tmp_path / "work"


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    """Fixture for cache directory"""
    return tmp_path / "cache"


@pytest.fixture
def test_subject(test_data_dir: Path) -> Dict[str, Path]:
    """Fixture for test subject data"""
    return create_test_subject(
        test_data_dir,
        "test_subject",
        shape=(64, 64, 64)
    )


@pytest.fixture
def mock_config(work_dir: Path, cache_dir: Path) -> PipelineConfig:
    """Fixture for mock pipeline configuration"""
    config = load_config()  # Load default config
    
    # Override paths
    config.working_dir = work_dir
    config.output_dir = work_dir / "output"
    config.cache.cache_dir = cache_dir
    
    # Set test-specific settings
    config.anthropic.api_key = "test-api-key"
    config.cache.max_cache_size_gb = 1.0
    config.cache.similarity_threshold = 0.8
    
    return config


@pytest.fixture
def pipeline(mock_config: PipelineConfig) -> Pipeline:
    """Fixture for pipeline instance"""
    return Pipeline(mock_config)


class TestPipeline:
    """Tests for pipeline orchestration"""
    
    @pytest.mark.asyncio
    async def test_initialization(self, pipeline: Pipeline):
        """Test pipeline initialization"""
        # Verify agent initialization
        assert "coordinator" in pipeline.agents
        assert "preprocessor" in pipeline.agents
        assert "analyzer" in pipeline.agents
        assert "visualizer" in pipeline.agents
        
        # Verify utility initialization
        assert pipeline.resource_monitor is not None
        assert pipeline.workflow_cache is not None
        assert pipeline.pipeline_monitor is not None
        
        # Verify directory creation
        assert pipeline.config.working_dir.exists()
        assert pipeline.config.output_dir.exists()
        assert pipeline.config.cache.cache_dir.exists()
    
    @pytest.mark.asyncio
    async def test_subject_processing(
        self,
        pipeline: Pipeline,
        test_subject: Dict[str, Path]
    ):
        """Test processing a single subject"""
        # Process subject
        result = await pipeline.process_subject(
            "test_subject",
            test_subject
        )
        
        # Verify successful completion
        assert result["status"] == "completed"
        assert "workflow_id" in result
        assert "metrics" in result
        assert "analysis_results" in result
        assert "visualization_paths" in result
        
        # Verify metrics
        metrics = result["metrics"]
        assert "total_time" in metrics
        assert "step_times" in metrics
        assert "resource_usage" in metrics
        
        # Verify resource monitoring
        resource_usage = metrics["resource_usage"]
        assert "cpu" in resource_usage
        assert "memory" in resource_usage
        assert "disk" in resource_usage
        
        # Verify workflow caching
        cache_file = pipeline.config.cache.cache_dir / "workflow_test_subject.json"
        assert cache_file.exists()
    
    @pytest.mark.asyncio
    async def test_dataset_processing(
        self,
        pipeline: Pipeline,
        test_data_dir: Path
    ):
        """Test processing multiple subjects"""
        # Create test dataset
        subjects = ["sub-01", "sub-02"]
        for subject_id in subjects:
            create_test_subject(
                test_data_dir / subject_id,
                subject_id,
                shape=(64, 64, 64)
            )
        
        # Process dataset
        results = await pipeline.process_dataset(test_data_dir)
        
        # Verify results for each subject
        for subject_id in subjects:
            assert subject_id in results
            subject_result = results[subject_id]
            assert subject_result["status"] == "completed"
            assert "workflow_id" in subject_result
            assert "metrics" in subject_result
            assert "analysis_results" in subject_result
            assert "visualization_paths" in subject_result
    
    @pytest.mark.asyncio
    async def test_error_handling(
        self,
        pipeline: Pipeline,
        test_subject: Dict[str, Path]
    ):
        """Test error handling"""
        # Try to process with missing files
        result = await pipeline.process_subject(
            "test_subject",
            {"T1": "nonexistent.nii.gz"}
        )
        
        # Verify error handling
        assert result["status"] == "failed"
        assert "error" in result
        assert "Missing required input files" in result["error"]
    
    @pytest.mark.asyncio
    async def test_workflow_caching(
        self,
        pipeline: Pipeline,
        test_subject: Dict[str, Path]
    ):
        """Test workflow caching and optimization"""
        # Process first subject
        result1 = await pipeline.process_subject(
            "subject1",
            test_subject
        )
        assert result1["status"] == "completed"
        
        # Process similar subject
        result2 = await pipeline.process_subject(
            "subject2",
            test_subject
        )
        assert result2["status"] == "completed"
        
        # Verify cache utilization
        metrics1 = result1["metrics"]
        metrics2 = result2["metrics"]
        assert metrics2["total_time"] <= metrics1["total_time"]  # Should be faster
    
    @pytest.mark.asyncio
    async def test_resource_management(
        self,
        pipeline: Pipeline,
        test_subject: Dict[str, Path]
    ):
        """Test resource management"""
        # Get initial resource usage
        usage = await pipeline.resource_monitor.get_usage()
        assert isinstance(usage, ResourceUsage)
        
        # Process subject
        result = await pipeline.process_subject(
            "test_subject",
            test_subject
        )
        
        # Verify resource tracking
        metrics = result["metrics"]
        resource_usage = metrics["resource_usage"]
        
        assert resource_usage["cpu"]["max"] > 0
        assert resource_usage["memory"]["max"] > 0
        assert resource_usage["disk"]["max"] > 0
        
        # Verify resource cleanup
        final_usage = await pipeline.resource_monitor.get_usage()
        assert abs(final_usage.cpu_percent - usage.cpu_percent) < 10
        assert abs(final_usage.memory_percent - usage.memory_percent) < 10
    
    @pytest.mark.asyncio
    async def test_workflow_optimization(
        self,
        pipeline: Pipeline,
        test_subject: Dict[str, Path]
    ):
        """Test workflow optimization"""
        # Create cached workflow pattern
        pattern = WorkflowPattern(
            steps=[
                {
                    "step_id": "preprocessing",
                    "tool": "fsl_preprocessor",
                    "parameters": {"method": "fast"},
                    "resources": {
                        "cpu_cores": 2,
                        "memory_gb": 4.0
                    }
                }
            ],
            metrics={
                "duration": 10.0,
                "accuracy": 0.95
            },
            parameters={"input_files": test_subject}
        )
        
        # Save pattern
        await pipeline.workflow_cache.save_pattern(
            "cached_subject",
            pattern
        )
        
        # Process new subject
        result = await pipeline.process_subject(
            "test_subject",
            test_subject
        )
        
        # Verify optimization
        assert result["status"] == "completed"
        metrics = result["metrics"]
        assert metrics["total_time"] > 0  # Should complete successfully
        
        # Verify pattern adaptation
        cache_file = pipeline.config.cache.cache_dir / "workflow_test_subject.json"
        assert cache_file.exists()
        
        # The new pattern should be different from the original
        new_pattern = WorkflowPattern.parse_file(cache_file)
        assert new_pattern.steps != pattern.steps  # Should be optimized