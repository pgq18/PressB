"""Select bundled model implementations without importing their dependencies.

Upstream module names are retained for checkpoint compatibility. A process may
use one source tree per family; importing another tree under the same module
names is rejected rather than silently combining implementations.
"""
from __future__ import annotations

import json
import os
import hashlib
from pathlib import Path
import sys
import threading
import importlib.metadata
import sysconfig


_FAMILIES = {
    "walloss": ("wall_x/serving/policy/wall_x_policy.py", ("wall_x", "openpi_client")),
    "vlajepa": ("starVLA/inference/piper_policy.py", ("starVLA",)),
}
_LOCK = threading.RLock()
_INSERTED = {}


def builtin_source_root(family):
    if family not in _FAMILIES:
        raise ValueError(f"Unknown model family {family!r}; choose {sorted(_FAMILIES)}")
    return Path(__file__).resolve().parent / family


def source_manifest(family, source_root=None):
    root = Path(source_root).expanduser().resolve() if source_root is not None else builtin_source_root(family)
    path = root / "source_manifest.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Model source manifest must be an object: {path}")
    return data


def source_tree_sha256(family, source_root=None):
    """Hash live implementation files, independently of Git and local paths."""
    root = Path(source_root).expanduser().resolve() if source_root is not None else builtin_source_root(family)
    digest = hashlib.sha256()
    count = 0
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if (not path.is_file() or path.suffix not in {".py", ".cu", ".cuh", ".h"}
                or any(part in {"tests", "docs", "__pycache__", ".git"} or part.startswith("build")
                       for part in relative.parts)):
            continue
        digest.update(relative.as_posix().encode() + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
        count += 1
    if not count:
        raise ValueError(f"No implementation files in {root}")
    return digest.hexdigest()


def activate_source(family, source_root=None, *, profile=None, extension_dir=None):
    """Activate a bundled implementation, or an explicit legacy source tree.

    This only adjusts Python import search paths. It neither installs packages
    nor loads weights, starts CUDA, compiles extensions or starts training.
    """
    builtin = builtin_source_root(family)
    root = Path(source_root).expanduser().resolve(strict=True) if source_root is not None else builtin
    marker, namespaces = _FAMILIES[family]
    if profile == "protocol" and family == "walloss":
        marker, namespaces = "packages/openpi_client/websocket_policy_server.py", ("openpi_client",)
    elif profile is not None:
        raise ValueError("Only the Wall-OSS protocol-only source profile is supported")
    if not (root / marker).is_file():
        raise ValueError(f"Incomplete {family} source tree: missing {root / marker}")
    with _LOCK:
        for name, module in tuple(sys.modules.items()):
            if module is None or not any(name == prefix or name.startswith(prefix + ".") for prefix in namespaces):
                continue
            locations = []
            if getattr(module, "__file__", None):
                locations.append(Path(module.__file__).resolve())
            locations.extend(Path(item).resolve() for item in getattr(module, "__path__", ()))
            if not locations or any(not item.is_relative_to(root) for item in locations):
                raise RuntimeError(f"{name} is already imported from another source tree; start a separate model process")
        paths = [str(root)]
        if family == "walloss" and (root / "packages").is_dir():
            paths.append(str(root / "packages"))
        selected_extension = extension_dir or (os.environ.get("PRESSB_WALLX_EXTENSION_DIR") if family == "walloss" else None)
        if family == "walloss" and profile is None and root == builtin and selected_extension is None:
            pointer = Path.home() / ".cache/pressb/walloss/extensions/current.json"
            if pointer.is_file():
                current = json.loads(pointer.read_text())
                try:
                    torch_version = importlib.metadata.version("torch")
                except importlib.metadata.PackageNotFoundError:
                    torch_version = None
                if (current.get("python_cache_tag") == sys.implementation.cache_tag
                        and current.get("python_soabi") == sysconfig.get_config_var("SOABI")
                        and current.get("torch_version") == torch_version):
                    candidate = Path(current["extension_path"]).expanduser().resolve(strict=True)
                    if (str(candidate.parent) != str(Path(current["extension_dir"]).expanduser().resolve())
                            or hashlib.sha256(candidate.read_bytes()).hexdigest() != current["extension_sha256"]):
                        raise ValueError("Cached Wall-OSS extension differs from its build receipt")
                    receipt = json.loads(Path(current["build_receipt"]).read_text())
                    source_hashes = receipt.get("source_sha256", {})
                    expected_sources = {str(path.relative_to(root)) for path in (root / "csrc").iterdir()
                                        if path.suffix in {".cu", ".h"}}
                    if set(source_hashes) != expected_sources or any(
                            hashlib.sha256((root / name).read_bytes()).hexdigest() != digest
                            for name, digest in source_hashes.items()):
                        raise ValueError("Cached Wall-OSS extension was built from different CUDA sources; rebuild it")
                    selected_extension = str(candidate.parent)
        if selected_extension is not None:
            if family != "walloss":
                raise ValueError("extension_dir only applies to Wall-OSS")
            directory = Path(selected_extension).expanduser().resolve(strict=True)
            if not directory.is_dir() or not list(directory.glob("wallx_csrc*.so")):
                raise ValueError(f"No built wallx_csrc extension in {directory}")
            loaded = sys.modules.get("wallx_csrc")
            if loaded is not None and (not getattr(loaded, "__file__", None) or
                    not Path(loaded.__file__).resolve().is_relative_to(directory)):
                raise RuntimeError("wallx_csrc is already imported from another extension directory")
            paths.insert(0, str(directory))
        for old in _INSERTED.get(family, ()):
            if old not in paths:
                while old in sys.path:
                    sys.path.remove(old)
        for path in reversed(paths):
            while path in sys.path:
                sys.path.remove(path)
            sys.path.insert(0, path)
        _INSERTED[family] = paths
    return root


__all__ = ["activate_source", "builtin_source_root", "source_manifest", "source_tree_sha256"]
