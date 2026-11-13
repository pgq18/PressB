#!/usr/bin/env python3
"""Task-local torchrun launcher for a host whose own DNS name cannot resolve.

Pass normal torchrun arguments, including --master-addr=127.0.0.1 for this
single-host run. Only this parent process avoids looking up its own hostname;
worker processes, other DNS queries and system configuration are unchanged.
"""
import socket

_hostname = socket.gethostname()
_original_getfqdn = socket.getfqdn


def _local_getfqdn(name=""):
    if not name or name in (_hostname, "0.0.0.0"):
        return "localhost"
    return _original_getfqdn(name)


if __name__ == "__main__":
    socket.getfqdn = _local_getfqdn
    from torch.distributed.run import main

    main()
