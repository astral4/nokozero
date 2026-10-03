"""Warp-section IDs and difficulties.

Named sections are `stage*1000 + category*100 + index` (category 1 = midboss/pre-boss, 2 = boss, in fight order).
Chapter warps are `10000 + stage*100 + portion`, with portion 1 being the start of the stage.
"""

from typing import Final

from nokozero.wire import CHAPTER_BASE, EXTRA_DIFFICULTY, EXTRA_STAGE, stage_of

LUNATIC: Final = 3


def opening(stage: int) -> int:
    """Return the section that a stage opens at."""
    return CHAPTER_BASE + stage * 100 + 1


def difficulty_for(section: int) -> int:
    """Return the difficulty to warp a section at."""
    return EXTRA_DIFFICULTY if stage_of(section) == EXTRA_STAGE else LUNATIC
