# Ray actor for managing the vllm rollouts.

import time
from vllm import AsyncEngineArgs, AsyncLLMEngine
from vllm.config import DeviceConfig, ModelConfig, VllmConfig
import ray


# problems with existing RL stacks
# 1. does not come with great observability tools out-of-the-box
# 2. it is complex, and meant to support many algorithms, it is not super hackable
# and easy to extend without extensive knowledge of Ray-distributed.

# - top-level performance
# - designed with ease-of-use from the ground up
# - designed with developer friendliness in mind -- blazing fast startup times, etc.



# 
# A light wrapper around the vllm engine.
# 
@ray.remote
class RolloutEngine:
    def __init__(self, model_name: str, num_workers: int):
        self.model_name = model_name
        self.num_workers = num_workers
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
