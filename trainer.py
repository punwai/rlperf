import ray

from config import Config

@ray.remote()
class Trainer:
    def __init__(self, config: Config):
        self.config = config

    def train(self):
        pass

    def evaluate(self):
        pass
