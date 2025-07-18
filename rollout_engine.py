# Ray actor for managing the vllm rollouts.

import time
from vllm import AsyncEngineArgs, AsyncLLMEngine
from vllm.config import DeviceConfig, ModelConfig, VllmConfig
import ray

from config import Config

# 
# A light wrapper around the vllm engine.
# 
@ray.remote
class RolloutEngine:
    def __init__(self, config: Config):
        self.model_name = config.model.name
        self.num_workers = 1
        self.engine = None
        self.engine_started = False

    def start_engine(self):
        print("Starting engine...")

        start_time = time.time()
        self.engine = AsyncLLMEngine.from_engine_args(
            engine_args=AsyncEngineArgs(
                model=self.model_name,
            )
        )
        end_time = time.time()
        print(f"Engine started in {end_time - start_time} seconds.")
        self.engine_started = True
        return True
    
    def stop_engine(self):
        """Clean shutdown of the engine"""
        if self.engine:
            print("Stopping vLLM engine...")
            # vLLM doesn't have explicit shutdown, but we can cleanup
            self.engine = None
            self.engine_started = False
        return True
    
    def get_status(self):
        """Get engine status for debugging"""
        return {
            "model_name": self.model_name,
            "num_workers": self.num_workers,
            "engine_started": self.engine_started,
            "engine_exists": self.engine is not None
        }
