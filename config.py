from pydantic import BaseModel, Field
from typing import Optional
import yaml
from pathlib import Path


class ValidatedBaseModel(BaseModel):
    def validate(self, config: dict):
        pass
    
    # will call validate on all children
    def tree_validate(self, config: dict):
        self.validate(config)
        for field_name, field_info in self.model_fields.items():
            field_value = getattr(self, field_name)
            if isinstance(field_value, ValidatedBaseModel):
                field_value.tree_validate(config[field_name])


class ExperimentConfig(BaseModel):
    # Wandb project name
    project_name: str = "default"
    # Experiment name for wandb & the name that the checkpoints will be saved under.
    name: str = "default"


class ModelConfig(BaseModel):
    name: str = "Qwen/Qwen3-0.6B"


# for vllm/sglang
class RolloutConfig(BaseModel):
    # The tensor parallel size for each model. The data parallel size
    tensor_parallel_size: int = 1
    # 
    max_input_length: int = 1024
    # 
    max_generation_length: int = 1024

# 
# The Trainer class is responsible for managing the PPO training loop.
# 

# how should we deal with multi-node training?

class TrainerConfig(BaseModel):
    # Number of nodes
    nnodes: int = 1
    # Number of GPUs/TPUs per node
    devices_per_node: int = 1
    # Whether to use activation checkpointing
    activation_checkpointing: bool = False

class Config(BaseModel):
    model: ModelConfig = ModelConfig()
    rollout: RolloutConfig = RolloutConfig()
    trainer: TrainerConfig = TrainerConfig()

    @classmethod
    def from_yaml(cls, path: str) -> 'Config':
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)

if __name__ == "__main__":
    config = Config.from_yaml("configs/default.yaml")
    print(config.model.name)