

# The scheduler is responsible for 

import ray
from config import Config
from ray.util.placement_group import (
    placement_group,
    placement_group_table,
    remove_placement_group,
)

class RayScheduler:
    """
    Manages the placement of workers onto compute.
    """
    def __init__(self, config: Config):
        self._init_inner(config.trainer.nnodes, config.trainer.devices_per_node)

    def _init_inner(self, nnodes: int, devices_per_node: int):
        self.nnodes = nnodes
        self.devices_per_node = devices_per_node
        self.pgs = [
            placement_group([{"CPU": 1} for _ in range(devices_per_node)])
            for _ in range(nnodes)
        ]

    def launch(self):
        """
        Given a ray remote function, perform gang scheduling onto the compute.
        """
        

