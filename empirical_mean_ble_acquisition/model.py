"""Empirical mean predictions; BLE is used only by the acquisition hook."""

from pathlib import Path
import sys

# Support both this source directory and the self-contained submission ZIP.
_ROOT = Path(__file__).resolve().parent
if not (_ROOT / "empirical_mean").is_dir():
    _ROOT = _ROOT.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from empirical_mean.model import predict
