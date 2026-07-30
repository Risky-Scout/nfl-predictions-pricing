"""2026 season operations: immutable prediction logging, live scoring, and the
live-promotion gate that governs when a PROVISIONAL candidate may stake."""

from nfl_hybrid.operations.season_loop import (
    LivePromotionConfig,
    log_week_predictions,
    score_resolved_week,
    cumulative_live_scorecard,
    live_promotion_decision,
)

__all__ = [
    "LivePromotionConfig",
    "log_week_predictions",
    "score_resolved_week",
    "cumulative_live_scorecard",
    "live_promotion_decision",
]
