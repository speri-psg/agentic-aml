"""Shared pytest configuration — adds project root to sys.path."""
import sys
import os

# Ensure project root is on the path before any test imports
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
