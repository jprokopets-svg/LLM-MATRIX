"""pytest configuration: add src/ to sys.path for llmmatrix package imports."""

import os
import sys

# Add src/ to path so `from llmmatrix import ...` works without pip install
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
