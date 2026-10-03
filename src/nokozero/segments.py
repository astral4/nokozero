"""Segment boundaries for episodes."""

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt


MIN_SEGMENT_FRAMES = 30


def crossed(previous: int, current: int) -> bool:
    """Whether the counter going from `previous` to `current` is a segment boundary."""
    return current < previous >= MIN_SEGMENT_FRAMES


def boundaries(time_in_chapter: npt.NDArray[np.integer]) -> npt.NDArray[np.intp]:
    """Return the segment boundary frames of a trace."""
    previous, current = time_in_chapter[:-1], time_in_chapter[1:]
    return np.flatnonzero((current < previous) & (previous >= MIN_SEGMENT_FRAMES)) + 1
