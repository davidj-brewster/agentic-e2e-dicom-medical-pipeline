"""
Test suite for ROI analysis functionality.
"""
import json
from pathlib import Path
from typing import Dict, List

import pytest
import numpy as np
import nibabel as nib

from utils.statistics import StatisticalComparison
from core.config import PipelineConfig
from cli.main import main


class TestROIAnalysis:
    """Tests for ROI analysis functionality"""
    
    @pytest.mark.asyncio
    async def test_roi_analysis(
        self,
        test_subjects: List[Dict[str, Path]],
        mock_config: PipelineConfig
    ):
        """Test ROI analysis"""
        # Create output directory
        output_dir = mock_config.output_dir / "roi_test"
        
        # Test with ROI masks
        args = [
            "compare-subjects",
            *[s["id"] for s in test_subjects],
            "--output", str(output_dir)
        ]
        exit_code = await main(args)
        assert exit_code == 0
        
        # Verify statistics
        with open(output_dir / "statistics.json") as f:
            stats = json.load(f)
        
        assert "roi_statistics" in stats
        for subject in test_subjects:
            assert subject["id"] in stats["roi_statistics"]
            subject_stats = stats["roi_statistics"][subject["id"]]
            assert "mean" in subject_stats
            assert "std" in subject_stats
            assert "volume" in subject_stats
            assert "skewness" in subject_stats
            assert "kurtosis" in subject_stats
            assert "iqr" in subject_stats
    
    @pytest.mark.asyncio
    async def test_roi_mask_handling(
        self,
        test_subjects: List[Dict[str, Path]],
        mock_config: PipelineConfig
    ):
        """Test ROI mask handling"""
        # Create test masks
        masks = []
        for subject in test_subjects:
            # Create random binary mask
            mask_data = np.random.randint(0, 2, (64, 64, 64))
            mask_img = nib.Nifti1Image(mask_data, np.eye(4))
            
            # Save mask
            mask_path = Path(subject["T1"]).parent / "mask.nii.gz"
            nib.save(mask_img, mask_path)
            masks.append(mask_path)
        
        # Test with masks
        args = [
            "compare-subjects",
            *[s["id"] for s in test_subjects],
            "--masks", *[str(m) for m in masks],
            "--output", str(mock_config.output_dir)
        ]
        exit_code = await main(args)
        assert exit_code == 0
        
        # Verify mask-specific statistics
        with open(mock_config.output_dir / "statistics.json") as f:
            stats = json.load(f)
        
        for subject in test_subjects:
            subject_stats = stats["roi_statistics"][subject["id"]]
            
            # Verify mask-based calculations
            assert subject_stats["volume"] > 0  # Should have some volume
            assert 0 <= subject_stats["mean"] <= 1  # Binary mask
            assert subject_stats["std"] >= 0
    
    @pytest.mark.asyncio
    async def test_roi_statistics_calculation(
        self,
        test_subjects: List[Dict[str, Path]],
        mock_config: PipelineConfig
    ):
        """Test ROI statistics calculation"""
        # Create comparison with known data
        data = []
        masks = []
        for _ in test_subjects:
            # Create data with known statistics
            subject_data = np.zeros((10, 10, 10))
            subject_data[2:8, 2:8, 2:8] = 1  # Create a cube of ones
            data.append(subject_data)
            
            # Create mask covering the cube
            mask = np.zeros((10, 10, 10))
            mask[2:8, 2:8, 2:8] = 1
            masks.append(mask)
        
        comparison = StatisticalComparison(
            data,
            masks,
            [s["id"] for s in test_subjects]
        )
        
        # Calculate statistics
        stats = comparison.calculate_roi_statistics()
        
        # Verify each subject's statistics
        for subject_id in stats:
            subject_stats = stats[subject_id]
            
            # Known values for cube of ones
            assert subject_stats["mean"] == 1.0
            assert subject_stats["std"] == 0.0
            assert subject_stats["volume"] == 216  # 6x6x6 cube
            assert subject_stats["skewness"] == 0.0  # Symmetric
            assert subject_stats["kurtosis"] == 0.0  # No outliers
            assert subject_stats["iqr"] == 0.0  # All values same
    
    @pytest.mark.asyncio
    async def test_roi_error_handling(
        self,
        test_subjects: List[Dict[str, Path]],
        mock_config: PipelineConfig
    ):
        """Test ROI error handling"""
        # Test with missing mask
        args = [
            "compare-subjects",
            *[s["id"] for s in test_subjects],
            "--masks", "/nonexistent/mask.nii.gz",
            "--output", str(mock_config.output_dir)
        ]
        exit_code = await main(args)
        assert exit_code == 1
        
        # Test with invalid mask dimensions
        # Create mask with wrong dimensions
        mask_data = np.random.randint(0, 2, (32, 32, 32))  # Different dimensions
        mask_img = nib.Nifti1Image(mask_data, np.eye(4))
        mask_path = mock_config.output_dir / "invalid_mask.nii.gz"
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(mask_img, mask_path)
        
        args = [
            "compare-subjects",
            *[s["id"] for s in test_subjects],
            "--masks", str(mask_path),
            "--output", str(mock_config.output_dir)
        ]
        exit_code = await main(args)
        assert exit_code == 1
        
        # Test with non-binary mask
        # Create non-binary mask
        mask_data = np.random.rand(64, 64, 64)  # Continuous values
        mask_img = nib.Nifti1Image(mask_data, np.eye(4))
        mask_path = mock_config.output_dir / "nonbinary_mask.nii.gz"
        nib.save(mask_img, mask_path)
        
        args = [
            "compare-subjects",
            *[s["id"] for s in test_subjects],
            "--masks", str(mask_path),
            "--output", str(mock_config.output_dir)
        ]
        exit_code = await main(args)
        assert exit_code == 1