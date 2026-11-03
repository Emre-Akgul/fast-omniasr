import sys
from pathlib import Path

# --import-mode=importlib doesn't put test directories on sys.path; shared test helpers
# (onnx_bytes) live beside the tests.
sys.path.insert(0, str(Path(__file__).parent))
