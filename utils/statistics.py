"""
Statistical comparison functionality.
Handles statistical analysis and visualization.
"""
import json
import logging
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from jinja2 import Template
from scipy import stats

logger = logging.getLogger(__name__)


class StatisticalTest(Enum):
    """Available statistical tests"""
    TTEST = auto()      # Independent t-test
    PAIRED = auto()     # Paired t-test
    ANOVA = auto()      # One-way ANOVA
    WILCOXON = auto()   # Wilcoxon rank-sum test


@dataclass
class StatisticalResult:
    """Statistical test result"""
    test_type: StatisticalTest
    statistic: float
    p_value: float
    effect_size: float
    confidence_interval: Optional[Tuple[float, float]] = None
    additional_metrics: Optional[Dict[str, float]] = None


class StatisticalComparison:
    """Statistical comparison system"""
    
    def __init__(
        self,
        images: List[np.ndarray],
        masks: Optional[List[np.ndarray]] = None,
        labels: Optional[List[str]] = None
    ):
        """
        Initialize comparison.
        
        Args:
            images: List of images to compare
            masks: Optional list of masks
            labels: Optional list of labels
        """
        self.images = images
        self.masks = masks
        self.labels = labels or [f"Image {i}" for i in range(len(images))]
        self.results: Dict[str, StatisticalResult] = {}
        
        # Validate inputs
        if len(self.images) < 2:
            raise ValueError("At least 2 images required for comparison")
        if masks and len(masks) != len(images):
            raise ValueError("Number of masks must match number of images")
        if labels and len(labels) != len(images):
            raise ValueError("Number of labels must match number of images")
        
        # Validate image shapes
        shape = images[0].shape
        if not all(img.shape == shape for img in images[1:]):
            raise ValueError("All images must have same shape")
    
    def calculate_roi_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate ROI statistics.
        
        Returns:
            Dictionary of statistics per image
        """
        stats = {}
        for i, image in enumerate(self.images):
            if self.masks:
                mask = self.masks[i]
                roi_data = image[mask > 0]
            else:
                roi_data = image.ravel()
            
            # Calculate basic statistics
            stats[self.labels[i]] = {
                "mean": float(np.mean(roi_data)),
                "std": float(np.std(roi_data)),
                "min": float(np.min(roi_data)),
                "max": float(np.max(roi_data)),
                "median": float(np.median(roi_data)),
                "volume": int(np.sum(self.masks[i])) if self.masks else int(roi_data.size)
            }
            
            # Calculate additional metrics
            stats[self.labels[i]].update({
                "skewness": float(stats.skew(roi_data)),
                "kurtosis": float(stats.kurtosis(roi_data)),
                "iqr": float(np.percentile(roi_data, 75) - np.percentile(roi_data, 25))
            })
        
        return stats
    
    def perform_statistical_test(
        self,
        test_type: StatisticalTest,
        alpha: float = 0.05
    ) -> StatisticalResult:
        """
        Perform statistical test.
        
        Args:
            test_type: Type of test to perform
            alpha: Significance level
            
        Returns:
            Statistical test result
        """
        if test_type == StatisticalTest.TTEST:
            if len(self.images) != 2:
                raise ValueError("T-test requires exactly 2 images")
            
            # Get data
            data1 = self.images[0][self.masks[0] > 0] if self.masks else self.images[0].ravel()
            data2 = self.images[1][self.masks[1] > 0] if self.masks else self.images[1].ravel()
            
            # Perform t-test
            t_stat, p_val = stats.ttest_ind(data1, data2)
            
            # Calculate effect size (Cohen's d)
            mean_diff = np.mean(data1) - np.mean(data2)
            pooled_std = np.sqrt(
                (np.std(data1)**2 + np.std(data2)**2) / 2
            )
            cohens_d = mean_diff / pooled_std
            
            # Calculate confidence interval
            ci = stats.t.interval(
                1 - alpha,
                len(data1) + len(data2) - 2,
                loc=mean_diff,
                scale=stats.sem(np.concatenate([data1, data2]))
            )
            
            result = StatisticalResult(
                test_type=test_type,
                statistic=float(t_stat),
                p_value=float(p_val),
                effect_size=float(cohens_d),
                confidence_interval=(float(ci[0]), float(ci[1]))
            )
            
        elif test_type == StatisticalTest.PAIRED:
            if len(self.images) != 2:
                raise ValueError("Paired t-test requires exactly 2 images")
            
            # Get data
            data1 = self.images[0][self.masks[0] > 0] if self.masks else self.images[0].ravel()
            data2 = self.images[1][self.masks[1] > 0] if self.masks else self.images[1].ravel()
            
            # Perform paired t-test
            t_stat, p_val = stats.ttest_rel(data1, data2)
            
            # Calculate effect size (Cohen's d for paired data)
            diff = data1 - data2
            cohens_d = np.mean(diff) / np.std(diff)
            
            # Calculate confidence interval
            ci = stats.t.interval(
                1 - alpha,
                len(diff) - 1,
                loc=np.mean(diff),
                scale=stats.sem(diff)
            )
            
            result = StatisticalResult(
                test_type=test_type,
                statistic=float(t_stat),
                p_value=float(p_val),
                effect_size=float(cohens_d),
                confidence_interval=(float(ci[0]), float(ci[1]))
            )
            
        elif test_type == StatisticalTest.ANOVA:
            # Get data
            data = [
                img[mask > 0] if self.masks else img.ravel()
                for img, mask in zip(
                    self.images,
                    self.masks if self.masks else [None] * len(self.images)
                )
            ]
            
            # Perform ANOVA
            f_stat, p_val = stats.f_oneway(*data)
            
            # Calculate effect size (eta-squared)
            groups = [np.full(len(d), i) for i, d in enumerate(data)]
            all_data = np.concatenate(data)
            all_groups = np.concatenate(groups)
            
            ss_between = sum(
                len(d) * (np.mean(d) - np.mean(all_data))**2
                for d in data
            )
            ss_total = sum(
                (x - np.mean(all_data))**2
                for x in all_data
            )
            eta_squared = ss_between / ss_total
            
            result = StatisticalResult(
                test_type=test_type,
                statistic=float(f_stat),
                p_value=float(p_val),
                effect_size=float(eta_squared)
            )
            
        else:  # WILCOXON
            if len(self.images) != 2:
                raise ValueError("Wilcoxon test requires exactly 2 images")
            
            # Get data
            data1 = self.images[0][self.masks[0] > 0] if self.masks else self.images[0].ravel()
            data2 = self.images[1][self.masks[1] > 0] if self.masks else self.images[1].ravel()
            
            # Perform Wilcoxon test
            stat, p_val = stats.ranksums(data1, data2)
            
            # Calculate effect size (r = Z / sqrt(N))
            n = len(data1) + len(data2)
            r = stat / np.sqrt(n)
            
            result = StatisticalResult(
                test_type=test_type,
                statistic=float(stat),
                p_value=float(p_val),
                effect_size=float(r)
            )
        
        # Store result
        self.results[test_type.name.lower()] = result
        return result
    
    def create_distribution_plot(self) -> Figure:
        """
        Create distribution plot.
        
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot distributions
        for i, image in enumerate(self.images):
            data = image[self.masks[i] > 0] if self.masks else image.ravel()
            sns.kdeplot(
                data=data,
                label=self.labels[i],
                ax=ax
            )
        
        # Add statistics
        stats = self.calculate_roi_statistics()
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.images)))
        for i, (label, stat) in enumerate(stats.items()):
            ax.axvline(
                stat["mean"],
                color=colors[i],
                linestyle='--',
                alpha=0.5
            )
        
        ax.set_xlabel("Intensity")
        ax.set_ylabel("Density")
        ax.set_title("Intensity Distributions")
        ax.legend()
        
        return fig
    
    def create_correlation_plot(self) -> Figure:
        """
        Create correlation plot.
        
        Returns:
            Matplotlib figure
        """
        if len(self.images) != 2:
            raise ValueError("Correlation plot requires exactly 2 images")
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Get data
        data1 = self.images[0][self.masks[0] > 0] if self.masks else self.images[0].ravel()
        data2 = self.images[1][self.masks[1] > 0] if self.masks else self.images[1].ravel()
        
        # Plot correlation
        ax.scatter(
            data1,
            data2,
            alpha=0.1,
            s=1
        )
        
        # Add correlation line
        z = np.polyfit(data1, data2, 1)
        p = np.poly1d(z)
        ax.plot(
            [np.min(data1), np.max(data1)],
            p([np.min(data1), np.max(data1)]),
            "r--",
            alpha=0.8
        )
        
        # Add correlation coefficient
        corr = np.corrcoef(data1, data2)[0, 1]
        ax.set_title(f"Correlation Plot (r = {corr:.3f})")
        
        ax.set_xlabel(self.labels[0])
        ax.set_ylabel(self.labels[1])
        
        return fig
    
    def save_results(self, output_dir: Path) -> None:
        """
        Save comparison results.
        
        Args:
            output_dir: Output directory
        """
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save statistics
        stats = {
            "roi_statistics": self.calculate_roi_statistics(),
            "test_results": {
                name: {
                    "test_type": result.test_type.name,
                    "statistic": result.statistic,
                    "p_value": result.p_value,
                    "effect_size": result.effect_size,
                    "confidence_interval": result.confidence_interval,
                    "additional_metrics": result.additional_metrics
                }
                for name, result in self.results.items()
            }
        }
        
        with open(output_dir / "statistics.json", "w") as f:
            json.dump(stats, f, indent=2)
        
        # Save plots
        self.create_distribution_plot().savefig(
            output_dir / "distributions.png"
        )
        
        if len(self.images) == 2:
            self.create_correlation_plot().savefig(
                output_dir / "correlation.png"
            )
        
        # Create HTML report
        template = Template("""
        <html>
        <head>
            <title>Statistical Comparison Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 2em; }
                h1, h2, h3 { color: #333; }
                .stats { margin: 1em 0; }
                .plot { margin: 2em 0; }
                .significant { color: #d32f2f; }
            </style>
        </head>
        <body>
            <h1>Statistical Comparison Report</h1>
            
            <h2>ROI Statistics</h2>
            <div class="stats">
            {% for label, stat in stats.roi_statistics.items() %}
                <h3>{{ label }}</h3>
                <ul>
                    <li>Mean: {{ "%.3f"|format(stat.mean) }}</li>
                    <li>Std: {{ "%.3f"|format(stat.std) }}</li>
                    <li>Median: {{ "%.3f"|format(stat.median) }}</li>
                    <li>Volume: {{ stat.volume }}</li>
                    <li>IQR: {{ "%.3f"|format(stat.iqr) }}</li>
                    <li>Skewness: {{ "%.3f"|format(stat.skewness) }}</li>
                    <li>Kurtosis: {{ "%.3f"|format(stat.kurtosis) }}</li>
                </ul>
            {% endfor %}
            </div>
            
            <h2>Statistical Tests</h2>
            <div class="stats">
            {% for name, result in stats.test_results.items() %}
                <h3>{{ name|title }}</h3>
                <ul>
                    <li>Test type: {{ result.test_type }}</li>
                    <li>Statistic: {{ "%.3f"|format(result.statistic) }}</li>
                    <li class="{{ 'significant' if result.p_value < 0.05 else '' }}">
                        P-value: {{ "%.3e"|format(result.p_value) }}
                    </li>
                    <li>Effect size: {{ "%.3f"|format(result.effect_size) }}</li>
                    {% if result.confidence_interval %}
                    <li>95% CI: ({{ "%.3f"|format(result.confidence_interval[0]) }},
                               {{ "%.3f"|format(result.confidence_interval[1]) }})</li>
                    {% endif %}
                </ul>
            {% endfor %}
            </div>
            
            <h2>Visualizations</h2>
            <div class="plot">
                <h3>Intensity Distributions</h3>
                <img src="distributions.png" alt="Distributions">
            </div>
            
            {% if correlation_plot %}
            <div class="plot">
                <h3>Correlation Plot</h3>
                <img src="correlation.png" alt="Correlation">
            </div>
            {% endif %}
        </body>
        </html>
        """)
        
        report_content = template.render(
            stats=stats,
            correlation_plot=len(self.images) == 2
        )
        with open(output_dir / "report.html", "w") as f:
            f.write(report_content)