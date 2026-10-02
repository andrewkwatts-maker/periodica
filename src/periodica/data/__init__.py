"""
Periodica data package
======================

Bundled data sheets (``active/``, ``defaults/``, ``derived/``,
``reference/``, ``config/``) plus the Python helpers that still read them
directly. Registry lookups belong to ``periodica.Get`` / ``periodica.data_sheet``;
this package keeps the user-data manager and the quark-source switch.

Names are exported lazily (PEP 562): ``import periodica.data`` loads no
submodule until one of these names is first used.

Use the unified DataManager for any category::

    from periodica.data import DataManager, DataCategory
    mgr = DataManager()
    items = mgr.get_all_items(DataCategory.AMINO_ACIDS)

Switch the quark data source (experimental <-> simulated)::

    from periodica.data import set_quark_source, SIMULATED
"""

import importlib

# public name -> submodule that defines it
_LAZY_EXPORTS = {
    # Data manager & categories
    "DataManager": "data_manager",
    "DataCategory": "data_manager",
    "get_data_manager": "data_manager",
    # Quark loader (used by the app's quark screen)
    "QuarkDataLoader": "quark_loader",
    "get_quark_loader": "quark_loader",
    # Quark data source switch (experimental <-> simulated/metaphysica)
    "EXPERIMENTAL": "quark_source",
    "SIMULATED": "quark_source",
    "get_quark_source": "quark_source",
    "set_quark_source": "quark_source",
    "load_quark": "quark_source",
    "list_quark_names": "quark_source",
    "iter_quarks": "quark_source",
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name):
    submodule = _LAZY_EXPORTS.get(name)
    if submodule is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f"{__name__}.{submodule}"), name)
    globals()[name] = value  # cache: later lookups skip __getattr__
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
