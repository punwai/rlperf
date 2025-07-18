import ray

from rlperf.config import Config
from rlperf.dataloader import create_dataloaders

@ray.remote
class Trainer:
    def __init__(self, config: Config):
        self.config = config
        self.train_loader, self.test_loader = create_dataloaders(config.dataset)

    def train(self):

        steps = 0
        for batch, ix in zip(self.train_loader, range(len(self.train_loader))):
            steps += 1
            print(steps)
            break

if __name__ == "__main__":
    config = Config.from_yaml("configs/default.yaml")

    trainer = Trainer.remote(config)
    ray.get(trainer.train.remote())