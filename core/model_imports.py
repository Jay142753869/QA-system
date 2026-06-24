"""Isolated import context for RE-GCN and TiRGN model packages.

Both model packages use identical module names (``rgcn``, ``src``) but have
incompatible implementations. This module provides a context manager that
temporarily swaps ``sys.path`` and ``sys.modules`` so each package can be
imported in isolation without cross-contamination.
"""

import sys
import threading
from contextlib import contextmanager

_import_lock = threading.Lock()


@contextmanager
def isolated_model_import(model_dir):
    """Context manager for isolated model package imports.

    Temporarily prepends *model_dir* to ``sys.path`` and removes any
    previously cached ``rgcn`` / ``src`` modules from ``sys.modules`` so
    that the target package's modules are imported fresh.  After the
    ``with`` block completes, the original ``sys.path`` and ``sys.modules``
    entries are restored.

    The caller must hold strong references to any imported module objects
    because they will no longer be reachable through ``sys.modules`` once
    the context exits.
    """
    with _import_lock:
        # --- snapshot current state ---
        original_path = sys.path[:]
        saved_modules = {}
        for key in list(sys.modules.keys()):
            if (
                key == "rgcn"
                or key.startswith("rgcn.")
                or key == "src"
                or key.startswith("src.")
            ):
                saved_modules[key] = sys.modules.pop(key)

        # --- set up for target package ---
        if model_dir in sys.path:
            sys.path.remove(model_dir)
        sys.path.insert(0, model_dir)

        try:
            yield
        finally:
            # --- remove everything loaded inside the context ---
            for key in list(sys.modules.keys()):
                if (
                    key == "rgcn"
                    or key.startswith("rgcn.")
                    or key == "src"
                    or key.startswith("src.")
                ):
                    del sys.modules[key]

            # --- restore original path ---
            sys.path[:] = original_path

            # --- restore previously cached modules ---
            sys.modules.update(saved_modules)
