import ray
import pynvml
import time
import psutil
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
from resources.scheduler import GangScheduler

@dataclass
class MemorySnapshot:
    """Single memory measurement snapshot"""
    timestamp: float
    node_id: str
    bundle_index: int
    gpu_id: int
    # GPU memory (MB)
    gpu_memory_used: float
    gpu_memory_total: float
    gpu_memory_percent: float
    # System memory (MB)
    system_memory_used: float
    system_memory_total: float
    system_memory_percent: float
    # GPU utilization
    gpu_utilization: float
    gpu_temperature: Optional[float] = None

@dataclass
class AggregatedStats:
    """Aggregated statistics across all loggers"""
    timestamp: float
    total_loggers: int
    # GPU stats
    avg_gpu_memory_percent: float
    max_gpu_memory_percent: float
    min_gpu_memory_percent: float
    total_gpu_memory_used: float
    total_gpu_memory_available: float
    # System stats  
    avg_system_memory_percent: float
    max_system_memory_percent: float
    avg_gpu_utilization: float
    max_gpu_utilization: float

@ray.remote
class ChildMemoryLogger:
    """Child logger that runs on each bundle to collect real-time memory stats"""
    
    def __init__(self):
        self.bundle_index = 0  # Hardcoded for now
        self.node_id = "0"    # Hardcoded for now
        
        # Initialize NVML for GPU monitoring
        try:
            pynvml.nvmlInit()
            self.gpu_available = True
            self.gpu_count = pynvml.nvmlDeviceGetCount()
        except Exception as e:
            self.gpu_available = False
            self.gpu_count = 0
    
    def get_current_memory_stats(self) -> MemorySnapshot:
        """Get real-time memory usage - all pynvml calls happen here"""
        timestamp = time.time()
        # Get system memory
        memory = psutil.virtual_memory()
        system_memory_used = memory.used / (1024**2)  # MB
        system_memory_total = memory.total / (1024**2)  # MB
        system_memory_percent = memory.percent
        # Get GPU memory (use first available GPU)
        gpu_memory_used = 0
        gpu_memory_total = 0
        gpu_memory_percent = 0
        gpu_utilization = 0
        gpu_temperature = None
        gpu_id = 0
        
        if self.gpu_available and self.gpu_count > 0:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_memory_used = mem_info.used / (1024**2)
                gpu_memory_total = mem_info.total / (1024**2)
                gpu_memory_percent = (mem_info.used / mem_info.total) * 100
                # Utilization - FRESH CALL
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_utilization = util.gpu
                # Temperature - FRESH CALL
                try:
                    gpu_temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                except:
                    gpu_temperature = None
            except Exception as e:
                print(f"Error getting GPU stats on bundle {self.bundle_index}: {e}")
        
        return MemorySnapshot(
            timestamp=timestamp,
            node_id=self.node_id,
            bundle_index=self.bundle_index,
            gpu_id=gpu_id,
            gpu_memory_used=gpu_memory_used,
            gpu_memory_total=gpu_memory_total,
            gpu_memory_percent=gpu_memory_percent,
            system_memory_used=system_memory_used,
            system_memory_total=system_memory_total,
            system_memory_percent=system_memory_percent,
            gpu_utilization=gpu_utilization,
            gpu_temperature=gpu_temperature
        )

@ray.remote
class CentralMemoryLogger:
    """Central coordinator that manages child loggers and aggregates stats"""
    
    def __init__(self, scheduler: GangScheduler):
        self.scheduler = scheduler
        self.child_loggers: List[ray.ObjectRef] = []
        self.monitoring_task = None
        self._setup_child_loggers()
    
    def _setup_child_loggers(self):
        """Create child loggers using gang scheduler"""
        self.child_loggers = self.scheduler.gang_schedule(ChildMemoryLogger, {})

    def _collect_stats_with_timeout(self, timeout: float = 5.0) -> List[MemorySnapshot]:
        """Collect stats from all children with timeout"""
        futures = [
            child.get_current_memory_stats.remote() 
            for child in self.child_loggers
        ]
        
        # Wait for results with timeout
        ready, not_ready = ray.wait(futures, timeout=timeout, num_returns=len(futures))
        
        if not_ready:
            print(f"Warning: {len(not_ready)} child loggers timed out")
            # Cancel timed out tasks
            for future in not_ready:
                ray.cancel(future)
        
        # Get results from ready tasks
        results = []
        for future in ready:
            try:
                results.append(ray.get(future))
            except Exception as e:
                print(f"Error getting result: {e}")
        
        return results

    def get_cluster_stats(self) -> AggregatedStats:
        """Get fresh cluster-wide memory stats with 5 second timeout"""
        snapshots = self._collect_stats_with_timeout(timeout=5.0)
        
        if not snapshots:
            print("No snapshots collected!")
            return None

        # Calculate aggregated statistics
        gpu_memory_percents = [s.gpu_memory_percent for s in snapshots if s.gpu_memory_percent > 0]
        system_memory_percents = [s.system_memory_percent for s in snapshots]
        gpu_utilizations = [s.gpu_utilization for s in snapshots if s.gpu_utilization >= 0]
        
        aggregated = AggregatedStats(
            timestamp=time.time(),
            total_loggers=len(snapshots),
            # GPU memory stats
            avg_gpu_memory_percent=sum(gpu_memory_percents) / len(gpu_memory_percents) if gpu_memory_percents else 0,
            max_gpu_memory_percent=max(gpu_memory_percents) if gpu_memory_percents else 0,
            min_gpu_memory_percent=min(gpu_memory_percents) if gpu_memory_percents else 0,
            total_gpu_memory_used=sum(s.gpu_memory_used for s in snapshots),
            total_gpu_memory_available=sum(s.gpu_memory_total for s in snapshots),
            # System memory stats
            avg_system_memory_percent=sum(system_memory_percents) / len(system_memory_percents) if system_memory_percents else 0,
            max_system_memory_percent=max(system_memory_percents) if system_memory_percents else 0,
            # GPU utilization
            avg_gpu_utilization=sum(gpu_utilizations) / len(gpu_utilizations) if gpu_utilizations else 0,
            max_gpu_utilization=max(gpu_utilizations) if gpu_utilizations else 0
        )
        
        return aggregated
    
    def start_continuous_monitoring(self, interval_seconds: float = 5.0, callback=None):
        """Start continuous monitoring in background"""
        if self.monitoring_task is not None:
            print("Monitoring already running")
            return
        
        @ray.remote
        def monitor_loop(logger_ref, interval, callback_fn):
            while True:
                try:
                    stats = ray.get(logger_ref.get_cluster_stats.remote())
                    if callback_fn and stats:
                        callback_fn(stats)
                    time.sleep(interval)
                except Exception as e:
                    print(f"Monitoring error: {e}")
                    time.sleep(interval)
        
        self.monitoring_task = monitor_loop.remote(self, interval_seconds, callback)
        print(f"Started continuous monitoring with {interval_seconds}s interval")
    
    def stop_continuous_monitoring(self):
        """Stop continuous monitoring"""
        if self.monitoring_task:
            ray.cancel(self.monitoring_task)
            self.monitoring_task = None
            print("Stopped continuous monitoring")
    
    def get_detailed_breakdown(self) -> List[MemorySnapshot]:
        """Get fresh detailed stats from each child logger"""
        return self._collect_stats_with_timeout(timeout=5.0)