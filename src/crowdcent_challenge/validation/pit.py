"""Point-in-time meta-model selection.

Model features may only use a meta snapshot published before the submission
date. Uniqueness scoring, by contrast, uses the meta-model built for the same
inference period after submissions close. These two purposes must not share an
implicit join rule.
"""

from collections.abc import Iterable
from datetime import date, timedelta
from enum import Enum

from .eras import HYPERLIQUID_RANKING_ERAS


class MetaPurpose(str, Enum):
    """Why a historical meta snapshot is being selected."""

    MODEL_INPUT = "model_input"
    UNIQUENESS_SCORING = "uniqueness_scoring"


def meta_release_cutoff(
    period: date,
    purpose: MetaPurpose,
    *,
    model_input_lag_days: int = 1,
) -> date:
    """Return the latest permissible meta release date.

    A one-day lag is the conservative default for model inputs because the
    current period's meta-model does not exist until submissions close.
    Uniqueness scoring uses the period's own meta-model.
    """

    if model_input_lag_days < 0:
        raise ValueError("model_input_lag_days must be non-negative")
    if purpose is MetaPurpose.MODEL_INPUT:
        return period - timedelta(days=model_input_lag_days)
    return period


def latest_available_meta_release(
    release_dates: Iterable[date],
    period: date,
    purpose: MetaPurpose,
    *,
    model_input_lag_days: int = 1,
) -> date | None:
    """Select the latest point-in-time meta release allowed for ``period``."""

    cutoff = meta_release_cutoff(
        period,
        purpose,
        model_input_lag_days=model_input_lag_days,
    )
    valid = [
        release
        for release in release_dates
        if HYPERLIQUID_RANKING_ERAS.has_meta_model(release) and release <= cutoff
    ]
    return max(valid, default=None)


def meta_coverage(
    periods: Iterable[date],
    release_dates: Iterable[date],
    purpose: MetaPurpose,
    *,
    model_input_lag_days: int = 1,
) -> float:
    """Fraction of periods with an admissible point-in-time meta snapshot."""

    period_list = list(periods)
    if not period_list:
        return 0.0
    releases = tuple(release_dates)
    covered = sum(
        latest_available_meta_release(
            releases,
            period,
            purpose,
            model_input_lag_days=model_input_lag_days,
        )
        is not None
        for period in period_list
    )
    return covered / len(period_list)
