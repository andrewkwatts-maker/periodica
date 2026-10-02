"""Python-side helpers that have not yet moved to the Rust core or the app.

Submodules are imported on demand (``from periodica.utils.bonding_rules
import ...``); importing this package loads nothing else. ``PhysicsConstants``
is still reachable here for backward compatibility and resolves lazily to
``periodica.physical_constants`` (PEP 562).
"""

__all__ = ["PhysicsConstants"]


def __getattr__(name):
    if name == "PhysicsConstants":
        from periodica.physical_constants import PhysicsConstants

        return PhysicsConstants
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
