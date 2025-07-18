# Ray actor for managing the vllm rollouts.

import time
import uuid
from transformers import AutoTokenizer
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
from vllm.config import DeviceConfig, ModelConfig, VllmConfig
import ray
import asyncio
from config import Config
from resources.scheduler import GangScheduler

# 
# A light wrapper around the vllm engine.
# 
@ray.remote
class RolloutEngine:
    def __init__(self, config: Config):
        self.model_name = config.model.name
        self.num_workers = 1
        self.engine = None
        self.config = config
        # use Huggingface tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def start_engine(self):
        self.engine = AsyncLLMEngine.from_engine_args(
            engine_args=AsyncEngineArgs(
                model=self.model_name,
                enforce_eager=True,
                distributed_executor_backend="ray"
            )
        )
    
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

    def sleep(self):
        asyncio.run(self.engine.sleep())
        return True
    
    def wake_up(self):
        """Wake up the engine"""
        asyncio.run(self.engine.wake_up())
        return True

    def generate(self, prompt: list[dict]):
        # Apply tokenizer to convert prompt to text format
        prompt_text = self.tokenizer.apply_chat_template(
            prompt, 
            tokenize=False, 
            add_generation_prompt=True
        )
        # vLLM returns AsyncGenerator[RequestOutput, None], so we need to capture all the individual
        # yielded outputs and return it all at once.
        async def _generate_async():
            async_generator = self.engine.generate(
                prompt_text,
                SamplingParams(
                    max_tokens=self.config.rollout.max_generation_length,
                ),
                str(uuid.uuid4())
            )
            outputs = []
            async for output in async_generator:
                outputs.append(output)
            return outputs[-1] if outputs else None
        
        return asyncio.run(_generate_async())

# What are you going to add?
# 1. Multi-modal support

if __name__ == "__main__":
    config = Config.from_yaml("configs/default.yaml")
    engine = RolloutEngine.remote(config)

    ray.get(engine.start_engine.remote())
    print(ray.get(engine.generate.remote([{"role": "user", "content": "Hello, how are you?"}])))