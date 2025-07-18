import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from rlperf.config import Config
from ray.util.placement_group import placement_group

class GangScheduler:
    """
    Manages the placement of workers onto compute.
    """
    def __init__(self, config: Config):
        self._init_inner(config.trainer.nnodes, config.trainer.devices_per_node)

    def _init_inner(self, nnodes: int, devices_per_node: int):
        self.nnodes = nnodes
        self.devices_per_node = devices_per_node
        self.pgs = [
            placement_group([{"CPU": 1, "GPU": 1} for _ in range(devices_per_node)])
            for _ in range(nnodes)
        ]
        ray.get([pg.ready() for pg in self.pgs])

    def gang_schedule(self, remote_cls, args):
        actors = []
        for pg in self.pgs:
            n_bundles = len(pg.bundle_specs)
            for _ in range(n_bundles):
                actor = remote_cls.options(
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=pg,
                    ),
                    num_gpus=0.001,
                    num_cpus=0.001,
                ).remote(args)
                actors.append(actor)

        return actors
