"""Test configuration: make `bikeslot` importable and keep test data out of the real data/ directory."""
import os
import sys
import tempfile
from pathlib import Path

_TMP = tempfile.mkdtemp(prefix="bikeslot-test-")
os.environ.setdefault("BIKESLOT_DATA_DIR", _TMP)
os.environ.setdefault("BIKESLOT_SEED_GOLDEN", "1")
os.environ.setdefault("BIKESLOT_PULL_CAMERAS", "")  # never poll the real camera from tests
os.environ.setdefault("BIKESLOT_DEMO_PREWARM", "0")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
