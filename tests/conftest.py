"""Put src/ on the import path so the tests can import the modules the same
way the scripts do (they run with src/ as the working folder)."""

import os
import sys

SRC = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, os.path.abspath(SRC))
