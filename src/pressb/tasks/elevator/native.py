"""Public elevator entry point backed by the common parallel simulation core."""
from pressb.runtime.parallel import SharedWorldBackend
from .configuration import load_config
from .parallel import ElevatorTask


class NativeElevatorBackend(SharedWorldBackend):
    def __init__(self, config, *, engine_factory=None):
        super().__init__(ElevatorTask(config), engine_factory=engine_factory)


def create_backend(config):
    return NativeElevatorBackend(config)
