import pathlib
import sys

# Ensure the repo root (this file's dir) is importable so `import vibench` works
# under pytest's prepend import mode without installing the package.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
