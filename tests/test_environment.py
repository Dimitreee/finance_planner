"""The scientific stack must actually work together, not merely be installed.

This exists because the machine's base conda environment has a `pandas` compiled against
NumPy 1.x running under NumPy 2.x, a `pyarrow` that fails to import, and no `scikit-learn`.
An import-only check would pass there; these round trips would not.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pyarrow  # noqa: F401  -- imported for its side effect of loading the Arrow libs


def test_pandas_and_numpy_interoperate() -> None:
    frame = pd.DataFrame({"value": np.arange(5, dtype="float64")})
    assert frame["value"].sum() == 10.0


def test_parquet_round_trip() -> None:
    frame = pd.DataFrame({"open_time": [0, 3_600_000], "close": [1.5, 2.5]})
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False)
    buffer.seek(0)
    assert pd.read_parquet(buffer).equals(frame)


def test_logistic_regression_fits() -> None:
    from sklearn.linear_model import LogisticRegression

    features = np.array([[0.0], [1.0], [2.0], [3.0]])
    labels = np.array([0, 0, 1, 1])
    model = LogisticRegression().fit(features, labels)
    assert model.predict_proba(features).shape == (4, 2)
