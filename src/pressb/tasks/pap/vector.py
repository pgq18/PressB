"""Historical PAP vector entry point using the common parallel runtime.

Scene/reset/reward hooks live in :mod:`.parallel`; this module owns no physics,
rendering, slot scheduling or independent episode execution loop.
"""
from pressb.runtime.parallel import SharedWorldBackend
from pressb.runtime.physics import path_rows as _path_rows
from pressb.runtime.isaac_parallel import encode_png as _encode_png
from .parallel import PAPTask, RESET_CONTRACT, load_config, renderer_profile


class PAPVectorBackend(SharedWorldBackend):
    def __init__(self, config, *, engine_factory=None):
        super().__init__(PAPTask(config), engine_factory=engine_factory)


def create_backend(config):
    return PAPVectorBackend(config)
