import sys
from pathlib import Path
# add project top-level to PYTHONPATH when something from this folder is imported
sys.path.append(str(Path(__path__[0]).parent))
