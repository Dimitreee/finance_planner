"""Shared domain core: data contracts, features, models, policy, explanation, backtest.

Terms used across this package are defined in CONTEXT.md at the repository root; decisions
that shaped them are recorded in docs/adr/.
"""

from __future__ import annotations

from importlib.metadata import version

__version__: str = version("cryptoguard")

__all__ = ["__version__"]
