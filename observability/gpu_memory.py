# 
# Ray Actor for monitoring GPU memory usage across the cluster.
# Logs to both WandB and a custom RLPerf viewer.
# 

import ray
import pynvml

# @ray.remote
# class GPUMemoryMonitor:
#     def __init__(self):
#         pynvml.nvmlInit()
#         self.handle = self.pynvml.nvmlDeviceGetHandleByIndex(0)
    
#     def run(self):
#         pass
