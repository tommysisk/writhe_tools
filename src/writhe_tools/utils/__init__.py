import os
import pkgutil
import importlib
import importlib.util

# Optional dependencies per module (no .py extension)
_optional_dependencies = {"graph_utils": ["torch_geometric"]}

# Mapping from modules to pip extra names
_optional_extras = {"graph_utils": "graph"}

_skipped_modules = []

def _has_all(modules):
    """Return True if all modules in the list are importable."""
    return all(importlib.util.find_spec(m) is not None for m in modules)

# Get the directory containing this file
__path__ = pkgutil.extend_path(__path__, __name__)
module_dir = os.path.dirname(__file__)

# Dynamically load all .py modules in this directory
for _, module_name, is_pkg in pkgutil.iter_modules([module_dir]):
    if is_pkg:
        continue  # Skip subdirectories (subpackages)

    # Check if this module requires optional dependencies
    required = _optional_dependencies.get(module_name, [])

    if required and not _has_all(required):
        missing = [m for m in required if importlib.util.find_spec(m) is None]
        extra = _optional_extras.get(module_name)
        print(f"[writhe_tools] ⏭️ Skipping '{module_name}' (missing: {', '.join(missing)})")
        if extra:
            print(f"[writhe_tools] 👉 To enable this module, install: pip install writhe-tools[{extra}]")
        _skipped_modules.append(module_name)
        continue

    # Import and expose all public symbols
    module = importlib.import_module(f".{module_name}", package=__name__)
    globals().update({
        name: getattr(module, name)
        for name in dir(module)
        if not name.startswith("_")
    })
