from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class BeliefState:
    particles: np.ndarray
    weights: np.ndarray
