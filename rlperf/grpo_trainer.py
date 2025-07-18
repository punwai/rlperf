from typing import Any
import ray
import torch
from rlperf.config import Config
from rlperf.rollout_engine import RolloutEngine

def main():
    config = Config.from_yaml("configs/default.yaml")
    # 
    rollout_engine: Any = RolloutEngine.remote(config)
    ray.get(rollout_engine.sleep.remote())

if __name__ == "__main__":
    main()
