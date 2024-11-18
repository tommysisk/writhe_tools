import os
import pkgutil
import importlib

# Get the directory containing this file
__path__ = pkgutil.extend_path(__path__, __name__)
module_dir = os.path.dirname(__file__)

# Dynamically load all modules in the current directory
for _, module_name, is_pkg in pkgutil.iter_modules([module_dir]):
    if not is_pkg:  # Ignore packages (subdirectories)
        module = importlib.import_module(f".{module_name}", package=__name__)
        globals().update({name: getattr(module, name) for name in dir(module) if not name.startswith("_")})