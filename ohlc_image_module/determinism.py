"""Utilities for enforcing deterministic behaviour across libraries."""
from __future__ import annotations

import os
import random

import numpy as np


def set_determinism(seed: int) -> None:
    """Set deterministic behaviour across supported libraries."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
