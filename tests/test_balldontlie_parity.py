"""Parity matrix invariants (Section 18, 25): nothing APPROXIMATE_NOT_APPROVED
or UNAVAILABLE may be treated as production-eligible."""
from __future__ import annotations

import pytest

from nfl_hybrid.providers.balldontlie.parity import (
    APPROXIMATE_NOT_APPROVED,
    EXACT,
    PARITY_MATRIX,
    UNAVAILABLE,
    VALIDATED_TRANSFORM,
    ParityIneligibleError,
    assert_production_eligible,
    by_family,
    non_eligible_families,
    production_eligible_families,
)


def test_every_entry_has_a_known_status():
    known = {EXACT, VALIDATED_TRANSFORM, APPROXIMATE_NOT_APPROVED, UNAVAILABLE}
    for entry in PARITY_MATRIX:
        assert entry.parity_status in known


def test_production_eligible_only_exact_or_validated_transform():
    for family in production_eligible_families():
        entry = by_family(family)
        assert entry.parity_status in (EXACT, VALIDATED_TRANSFORM)


def test_epa_family_is_unavailable_and_ineligible():
    entry = by_family("epa_success_cpoe (team + QB)")
    assert entry.parity_status == UNAVAILABLE
    assert entry.production_eligible is False


def test_success_rate_is_unavailable():
    entry = by_family("success_rate")
    assert entry.parity_status == UNAVAILABLE


def test_game_result_and_elo_are_eligible():
    assert by_family("game_result").production_eligible is True
    assert by_family("elo_inputs").production_eligible is True


def test_assert_production_eligible_raises_for_ineligible_family():
    for family in non_eligible_families():
        with pytest.raises(ParityIneligibleError):
            assert_production_eligible(family)


def test_assert_production_eligible_passes_for_eligible_family():
    entry = assert_production_eligible("game_result")
    assert entry.feature_family == "game_result"


def test_unknown_family_raises_keyerror():
    with pytest.raises(KeyError):
        by_family("not_a_real_family")


def test_no_duplicate_family_names():
    names = [e.feature_family for e in PARITY_MATRIX]
    assert len(names) == len(set(names))
