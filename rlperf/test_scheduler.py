import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from vllm.config import ModelConfig, VllmConfig
from rlperf.config import Config
from rlperf.mock.mock_worker import MockWorker, MockWorkerArgs
from rlperf.resources.scheduler import GangScheduler
from typing import TYPE_CHECKING
from vllm import AsyncLLMEngine

from rlperf.rollout_engine import RolloutEngine

if __name__ == "__main__":
    config = Config.from_yaml("configs/default.yaml")
    ray.init()

    # the rollout engine should not be gang-scheduled, as the memory is managed
    # separately. (i.e. vLLM manages it's own scheduling, and RLPerf manages it's own scheduling)
    rollout_engine = RolloutEngine.remote(config=config)
    generation = ray.get(rollout_engine.start_engine.remote())
    
    print(generation)

    # 
    # 
    # 
    # 
    # 
    # 
    # 
    