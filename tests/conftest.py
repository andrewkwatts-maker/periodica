"""Pytest configuration for periodica library tests."""
import sys
import os

# Ensure the library src is on the path for editable installs
lib_src = os.path.join(os.path.dirname(__file__), "..", "src")
if lib_src not in sys.path:
    sys.path.insert(0, os.path.abspath(lib_src))
