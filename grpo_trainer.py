import torch
from config import Config
from rollout_engine import RolloutEngine

def main():
    config = Config.from_yaml("configs/default.yaml")
    # 
    rollout_engine = RolloutEngine.remote(config)

if __name__ == "__main__":
    main()
