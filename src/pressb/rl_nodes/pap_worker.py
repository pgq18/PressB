"""Compatibility alias for :mod:`pressb.tasks.pap.worker`."""
import sys
from pressb.tasks.pap import worker as _implementation

if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
