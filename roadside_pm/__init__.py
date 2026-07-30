"""Checkout-time compatibility package for the ``src`` layout.

This makes ``python -m pipelines...`` work from the repository root before an
editable install. Installed environments continue to use ``src/roadside_pm``.
"""

from pathlib import Path


_SOURCE_PACKAGE = Path(__file__).resolve().parents[1] / "src" / "roadside_pm"
__path__ = [str(_SOURCE_PACKAGE)]

__version__ = "0.1.0"

