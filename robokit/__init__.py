"""
Top-level package for RoboKit.

Submodules are imported lazily to avoid pulling heavy optional dependencies
unless the user actually touches those namespaces.
"""

from importlib import import_module

__all__ = ["perception", "utils", "evaluation", "datasets", "ros", "iteach_toolkit"]


def __getattr__(name):
    if name in __all__:
        module = import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals().keys()) | set(__all__))
