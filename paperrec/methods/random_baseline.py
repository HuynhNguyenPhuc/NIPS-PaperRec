"""Random baseline: K catalogue papers drawn uniformly, ignoring the query."""

from __future__ import annotations

import numpy as np

from paperrec.catalogue import Catalogue
from paperrec.config import K


def recommend_random(catalogue: Catalogue, rng: np.random.Generator) -> list[int]:
    """
    Draw K catalogue papers uniformly at random.

    Args:
        catalogue: Indexed catalogue whose size bounds the draw.
        rng: Random generator; its state is advanced by the draw.

    Returns:
        positions: K distinct catalogue positions.
    """
    return list(rng.choice(len(catalogue.ids), size=K, replace=False))
