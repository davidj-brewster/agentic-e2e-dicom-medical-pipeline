"""
Utilities for pipeline orchestration.
Provides workflow caching, resource management, and monitoring.
"""
import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import anthropic
import psutil
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


@dataclass
class ResourceUsage:
    """Current resource usage"""
    cpu_percent: float
    memory_percent: float
    disk_usage: float
    gpu_memory_used: Optional[float] = None


class WorkflowPattern(BaseModel):
    """Pattern extracted from successful workflow"""
    steps: List[Dict[str, Any]]
    metrics: Dict[str, float]
    parameters: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class CacheEntry(BaseModel):
    """Cache entry for workflow optimization"""
    subject_id: str
    pattern: WorkflowPattern
    prompt_templates: Dict[str, str]
    similarity_score: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ResourceMonitor:
    """Monitors system resource usage"""

    def __init__(self, gpu_enabled: bool = False):
        self.gpu_enabled = gpu_enabled
        if gpu_enabled:
            try:
                import torch
                self.torch = torch
            except ImportError:
                logger.warning("GPU monitoring enabled but PyTorch not available")
                self.gpu_enabled = False

    async def get_usage(self) -> ResourceUsage:
        """Get current resource usage"""
        try:
            # Get CPU and memory usage
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")
            
            # Get GPU memory if enabled
            gpu_memory = None
            if self.gpu_enabled:
                try:
                    gpu_memory = self.torch.cuda.memory_allocated() / 1024**3  # GB
                except Exception as e:
                    logger.warning(f"Error getting GPU memory: {e}")
            
            return ResourceUsage(
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                disk_usage=disk.percent,
                gpu_memory_used=gpu_memory
            )
            
        except Exception as e:
            logger.error(f"Error getting resource usage: {e}")
            return ResourceUsage(
                cpu_percent=0.0,
                memory_percent=0.0,
                disk_usage=0.0
            )

    async def check_resources(
        self,
        required_cpu: float,
        required_memory: float,
        required_disk: float,
        required_gpu: Optional[float] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if required resources are available.
        
        Args:
            required_cpu: Required CPU cores
            required_memory: Required memory in GB
            required_disk: Required disk space in GB
            required_gpu: Required GPU memory in GB
            
        Returns:
            Tuple of (resources_available, error_message)
        """
        try:
            usage = await self.get_usage()
            
            # Check CPU
            if usage.cpu_percent > 90:  # Leave some headroom
                return False, "CPU usage too high"
            
            # Check memory
            memory = psutil.virtual_memory()
            available_memory = memory.available / 1024**3  # GB
            if available_memory < required_memory:
                return False, f"Insufficient memory: need {required_memory}GB, have {available_memory:.1f}GB"
            
            # Check disk
            disk = psutil.disk_usage("/")
            available_disk = disk.free / 1024**3  # GB
            if available_disk < required_disk:
                return False, f"Insufficient disk space: need {required_disk}GB, have {available_disk:.1f}GB"
            
            # Check GPU if required
            if required_gpu and self.gpu_enabled:
                try:
                    total = self.torch.cuda.get_device_properties(0).total_memory / 1024**3
                    used = usage.gpu_memory_used or 0
                    available = total - used
                    if available < required_gpu:
                        return False, f"Insufficient GPU memory: need {required_gpu}GB, have {available:.1f}GB"
                except Exception as e:
                    logger.warning(f"Error checking GPU memory: {e}")
                    if required_gpu > 0:
                        return False, "GPU memory check failed"
            
            return True, None
            
        except Exception as e:
            logger.error(f"Error checking resources: {e}")
            return False, str(e)


class WorkflowCache:
    """Manages workflow caching and optimization"""

    def __init__(
        self,
        cache_dir: Path,
        anthropic_client: anthropic.Client,
        max_cache_size: float = 10.0,  # GB
        similarity_threshold: float = 0.8
    ):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.client = anthropic_client
        self.max_cache_size = max_cache_size
        self.similarity_threshold = similarity_threshold

    async def save_pattern(
        self,
        subject_id: str,
        pattern: WorkflowPattern
    ) -> None:
        """Save workflow pattern to cache"""
        try:
            # Generate prompt templates
            prompt_templates = {}
            
            for step in pattern.steps:
                prompt = f"""
                Task: Process neuroimaging data for subject analysis
                Step: {step['step_id']}
                Tool: {step['tool']}
                Parameters: {json.dumps(step['parameters'], indent=2)}
                Metrics: {json.dumps(step.get('metrics', {}), indent=2)}
                
                Generate a template for processing similar subjects.
                The template should capture the key parameters and decision points
                while allowing for subject-specific adaptation.
                """
                
                response = await asyncio.to_thread(
                    self.client.messages.create,
                    model="claude-3-sonnet-20240229",
                    max_tokens=1000,
                    messages=[{
                        "role": "user",
                        "content": prompt
                    }]
                )
                
                prompt_templates[step['step_id']] = response.content
            
            # Create cache entry
            entry = CacheEntry(
                subject_id=subject_id,
                pattern=pattern,
                prompt_templates=prompt_templates,
                similarity_score=1.0  # Perfect match for own subject
            )
            
            # Save to file
            cache_path = self.cache_dir / f"workflow_{subject_id}.json"
            cache_path.write_text(entry.json(indent=2))
            
            # Clean up old cache entries if needed
            await self._cleanup_cache()
            
        except Exception as e:
            logger.error(f"Error saving workflow pattern: {e}")

    async def find_similar_pattern(
        self,
        subject_id: str,
        subject_data: Dict[str, Any]
    ) -> Optional[CacheEntry]:
        """Find similar workflow pattern in cache"""
        try:
            # Get all cache entries
            cache_files = list(self.cache_dir.glob("workflow_*.json"))
            if not cache_files:
                return None
            
            best_match = None
            best_score = 0.0
            
            for cache_file in cache_files:
                try:
                    entry = CacheEntry.parse_file(cache_file)
                    
                    # Calculate similarity score
                    prompt = f"""
                    Previous subject data:
                    {json.dumps(entry.pattern.parameters, indent=2)}
                    
                    Current subject data:
                    {json.dumps(subject_data, indent=2)}
                    
                    Calculate a similarity score between 0.0 and 1.0 for these subjects.
                    Consider data characteristics, processing requirements, and quality metrics.
                    Return only the numeric score.
                    """
                    
                    response = await asyncio.to_thread(
                        self.client.messages.create,
                        model="claude-3-sonnet-20240229",
                        max_tokens=100,
                        messages=[{
                            "role": "user",
                            "content": prompt
                        }]
                    )
                    
                    try:
                        score = float(response.content)
                        if score > best_score:
                            best_score = score
                            best_match = entry
                    except ValueError:
                        continue
                        
                except Exception as e:
                    logger.warning(f"Error processing cache entry {cache_file}: {e}")
                    continue
            
            if best_score >= self.similarity_threshold:
                best_match.similarity_score = best_score
                return best_match
            
            return None
            
        except Exception as e:
            logger.error(f"Error finding similar pattern: {e}")
            return None

    async def optimize_workflow(
        self,
        pattern: WorkflowPattern,
        subject_data: Dict[str, Any],
        prompt_templates: Dict[str, str]
    ) -> WorkflowPattern:
        """Optimize workflow pattern for current subject"""
        try:
            optimized_steps = []
            
            for step in pattern.steps:
                template = prompt_templates.get(step['step_id'])
                if not template:
                    optimized_steps.append(step)
                    continue
                
                prompt = f"""
                Previous workflow step:
                {json.dumps(step, indent=2)}
                
                Template:
                {template}
                
                Current subject data:
                {json.dumps(subject_data, indent=2)}
                
                Optimize this workflow step for the current subject.
                Consider data characteristics, resource requirements, and quality metrics.
                Return only the optimized step configuration as valid JSON.
                """
                
                response = await asyncio.to_thread(
                    self.client.messages.create,
                    model="claude-3-sonnet-20240229",
                    max_tokens=2000,
                    messages=[{
                        "role": "user",
                        "content": prompt
                    }]
                )
                
                try:
                    optimized_step = json.loads(response.content)
                    optimized_steps.append(optimized_step)
                except json.JSONDecodeError:
                    optimized_steps.append(step)
            
            return WorkflowPattern(
                steps=optimized_steps,
                metrics=pattern.metrics,
                parameters=pattern.parameters
            )
            
        except Exception as e:
            logger.error(f"Error optimizing workflow: {e}")
            return pattern

    async def _cleanup_cache(self) -> None:
        """Clean up old cache entries if size limit exceeded"""
        try:
            # Get cache size
            cache_size = sum(f.stat().st_size for f in self.cache_dir.glob("*.json"))
            cache_size_gb = cache_size / 1024**3
            
            if cache_size_gb > self.max_cache_size:
                # Remove oldest entries until under limit
                cache_files = sorted(
                    self.cache_dir.glob("*.json"),
                    key=lambda p: p.stat().st_mtime
                )
                
                while cache_size_gb > self.max_cache_size and cache_files:
                    file_to_remove = cache_files.pop(0)
                    cache_size_gb -= file_to_remove.stat().st_size / 1024**3
                    file_to_remove.unlink()
                    
        except Exception as e:
            logger.error(f"Error cleaning up cache: {e}")


class PipelineMonitor:
    """Monitors pipeline execution and resource usage"""

    def __init__(self, resource_monitor: ResourceMonitor):
        self.resource_monitor = resource_monitor
        self.start_time = datetime.utcnow()
        self.step_times: Dict[str, float] = {}
        self.resource_usage: List[ResourceUsage] = []
        self.sampling_interval = 5  # seconds

    async def start_monitoring(self) -> None:
        """Start resource monitoring"""
        self.start_time = datetime.utcnow()
        self.monitoring = True
        
        while self.monitoring:
            usage = await self.resource_monitor.get_usage()
            self.resource_usage.append(usage)
            await asyncio.sleep(self.sampling_interval)

    async def stop_monitoring(self) -> None:
        """Stop resource monitoring"""
        self.monitoring = False
        self.end_time = datetime.utcnow()

    def record_step_time(self, step_id: str, duration: float) -> None:
        """Record execution time for a step"""
        self.step_times[step_id] = duration

    def get_metrics(self) -> Dict[str, Any]:
        """Get execution metrics"""
        total_time = (datetime.utcnow() - self.start_time).total_seconds()
        
        # Calculate resource usage statistics
        cpu_usage = [u.cpu_percent for u in self.resource_usage]
        memory_usage = [u.memory_percent for u in self.resource_usage]
        disk_usage = [u.disk_usage for u in self.resource_usage]
        gpu_usage = [u.gpu_memory_used for u in self.resource_usage if u.gpu_memory_used]
        
        return {
            "total_time": total_time,
            "step_times": self.step_times,
            "resource_usage": {
                "cpu": {
                    "mean": sum(cpu_usage) / len(cpu_usage),
                    "max": max(cpu_usage),
                    "min": min(cpu_usage)
                },
                "memory": {
                    "mean": sum(memory_usage) / len(memory_usage),
                    "max": max(memory_usage),
                    "min": min(memory_usage)
                },
                "disk": {
                    "mean": sum(disk_usage) / len(disk_usage),
                    "max": max(disk_usage),
                    "min": min(disk_usage)
                },
                "gpu": {
                    "mean": sum(gpu_usage) / len(gpu_usage) if gpu_usage else None,
                    "max": max(gpu_usage) if gpu_usage else None,
                    "min": min(gpu_usage) if gpu_usage else None
                }
            }
        }