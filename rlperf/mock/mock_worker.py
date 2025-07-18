from dataclasses import dataclass
import time
import ray
import random


@dataclass
class MockWorkerArgs:
    x: int

@ray.remote(
    num_gpus=0.01,
    num_cpus=0.01,
)
class MockWorker:
    def __init__(self, args: MockWorkerArgs):
        self.id = random.randint(0, 1000000)
        print(f"Mock worker {self.id} initialized")
        print(f"x: {args.x}")
        pass
    
    def run(self):
        while True:
            time.sleep(1)
            print(f"Mock worker {self.id} running {ray.get_gpu_ids()}")