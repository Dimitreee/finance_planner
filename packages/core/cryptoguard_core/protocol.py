"""The decision protocol's fixed instants, in one place.

These are the values ADR-0001 froze. They live here rather than in the dataset module so that the
read API can state them to a reader without importing the data path, and rather than being read
from the Experiment Contract at serve time so that the API holds no configuration beyond its
database. The contract loader checks the contract against them, so the two cannot drift.
"""

from __future__ import annotations

FEATURE_CUTOFF_UTC = "00:00"
DECISION_DEADLINE_UTC = "00:10"
EXECUTION_UTC = "01:00"
HORIZON_DAYS = 1
