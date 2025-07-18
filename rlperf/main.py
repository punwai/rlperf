import torch
from rlperf.config import Config
from rlperf.rollout_engine import RolloutEngine

def main():
    config = Config.from_yaml("config.yaml")
    RolloutEngine.remote(config)

if __name__ == "__main__":
    main()
