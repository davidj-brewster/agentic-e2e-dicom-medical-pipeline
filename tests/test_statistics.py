"""
Test suite for statistical analysis functionality.
"""
import json
from pathlib import Path
from typing import Dict, List

import pytest
import numpy as np

from utils.statistics import StatisticalTest, StatisticalComparison
from core.config import PipelineConfig
from cli.main import main


class TestStatisticalAnalysis:
    """Tests for statistical analysis functionality"""
    
    @pytest.mark.asyncio
    async def test_statistical_analysis(
        self,
        test_subjects: List[Dict[str, Path]],
        mock_config: PipelineConfig
    ):
        """Test statistical analysis"""
        # Create output directory
        output_dir = mock_config.output_dir / "statistical_test"
        
        # Test with different statistical tests
        for test in StatisticalTest:
            args = [
                "compare-subjects",
                *[s["id"] for s in test_subjects],
                "--test", test.name.lower(),
                "--output", str(output_dir)
            ]
            exit_code = await main(args)
            assert exit_code == 0
            
            # Verify outputs
            assert (output_dir / "statistics.json").exists()
            assert (output_dir / "distributions.png").exists()
            assert (output_dir / "correlation.png").exists()
            assert (output_dir / "report.html").exists()
            
            # Verify statistics
            with open(output_dir / "statistics.json") as f:
                stats = json.load(f)
            
            assert "roi_statistics" in stats
            assert "test_results" in stats
            assert test.name.lower() in stats["test_results"]
            
            result = stats["test_results"][test.name.lower()]
            assert "statistic" in result
            assert "p_value" in result
            assert "effect_size" in result
            
            # Verify p-value is valid
            assert 0 <= result["p_value"] <= 1
            
            # Verify effect size is valid
            assert isinstance(result["effect_size"], float)
            
            # Verify confidence intervals if available
            if "confidence_interval" in result:
                ci_low, ci_high = result["confidence_interval"]
                assert ci_low < ci_high
    
    @pytest.mark.asyncio
    async def test_statistical_comparison(
        self,
        test_subjects: List[Dict[str, Path]],
        mock_config: PipelineConfig
    ):
        """Test statistical comparison class"""
        # Create comparison object
        comparison = StatisticalComparison(
            [np.random.rand(10, 10, 10) for _ in test_subjects],
            None,  # No masks
            [s["id"] for s in test_subjects]
        )
        
        # Test each statistical test
        for test in StatisticalTest:
            result = comparison.perform_statistical_test(test)
            
            assert result.test_type == test
            assert isinstance(result.statistic, float)
            assert 0 <= result.p_value <= 1
            assert isinstance(result.effect_size, float)
            
            if result.confidence_interval:
                ci_low, ci_high = result.confidence_interval
                assert ci_low < ci_high
    
    @pytest.mark.asyncio
    async def test_roi_statistics(
        self,
        test_subjects: List[Dict[str, Path]],
        mock_config: PipelineConfig
    ):
        """Test ROI statistics calculation"""
        # Create comparison object with masks
        masks = [np.random.randint(0, 2, (10, 10, 10)) for _ in test_subjects]
        comparison = StatisticalComparison(
            [np.random.rand(10, 10, 10) for _ in test_subjects],
            masks,
            [s["id"] for s in test_subjects]
        )
        
        # Calculate ROI statistics
        stats = comparison.calculate_roi_statistics()
        
        # Verify statistics for each subject
        for subject_id in [s["id"] for s in test_subjects]:
            assert subject_id in stats
            subject_stats = stats[subject_id]
            
            # Check required statistics
            assert "mean" in subject_stats
            assert "std" in subject_stats
            assert "volume" in subject_stats
            assert "skewness" in subject_stats
            assert "kurtosis" in subject_stats
            assert "iqr" in subject_stats
            
            # Verify values are valid
            assert isinstance(subject_stats["mean"], float)
            assert subject_stats["std"] >= 0
            assert subject_stats["volume"] >= 0
            assert isinstance(subject_stats["skewness"], float)
            assert isinstance(subject_stats["kurtosis"], float)
            assert subject_stats["iqr"] >= 0
    
    @pytest.mark.asyncio
    async def test_error_handling(
        self,
        test_subjects: List[Dict[str, Path]],
        mock_config: PipelineConfig
    ):
        """Test error handling in statistical analysis"""
        # Test with invalid test type
        args = [
            "compare-subjects",
            *[s["id"] for s in test_subjects],
            "--test", "invalid_test",
            "--output", str(mock_config.output_dir)
        ]
        exit_code = await main(args)
        assert exit_code == 1
        
        # Test with mismatched dimensions
        comparison = StatisticalComparison(
            [np.random.rand(10, 10, 10), np.random.rand(20, 20, 20)],
            None,
            ["sub-01", "sub-02"]
        )
        
        with pytest.raises(ValueError):
            comparison.perform_statistical_test(StatisticalTest.TTEST)
        
        # Test with invalid masks
        with pytest.raises(ValueError):
            StatisticalComparison(
                [np.random.rand(10, 10, 10) for _ in range(2)],
                [np.random.rand(20, 20, 20) for _ in range(2)],  # Mismatched dimensions
                ["sub-01", "sub-02"]
            )