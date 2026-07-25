from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EnsembleInputs:
    yahoo: float
    five_thirty_eight: float
    vegas: float


def _clamp(p: float) -> float:
    return min(max(float(p), 0.0), 1.0)


def _favorite_probability(p: float) -> float:
    return max(p, 1.0 - p)


def _hard_pick(p: float) -> float:
    return 1.0 if p > 0.5 else 0.0 if p < 0.5 else 0.5


def _round_if(values: dict[int, float], round_output: bool) -> dict[int, float]:
    if not round_output:
        return {k: _clamp(v) for k, v in values.items()}
    return {k: round(_clamp(v), 2) for k, v in values.items()}


def legacy_ensemble_models(
    inputs: EnsembleInputs,
    *,
    round_output: bool = False,
) -> dict[int, float]:
    """Logical single-team translations of all 32 workbook ensemble models.

    The workbook stores each game as paired team rows and uses cross-row
    complements. This function returns the probability for the team represented
    by the three supplied probabilities. Model 4 simplifies to FiveThirtyEight
    under the workbook's documented interpretation.
    """
    y = _clamp(inputs.yahoo)
    f = _clamp(inputs.five_thirty_eight)
    v = _clamp(inputs.vegas)

    agree_yf = (y - 0.5) * (f - 0.5) >= 0
    diff_yf = abs(y - f)
    diff_fv = abs(f - v)

    m1 = (y + f) / 2.0
    m2 = (f + 0.5) / 2.0
    m3 = f if agree_yf else 0.5
    m4 = f

    factors = {
        5: 1.05,
        6: 1.10,
        7: 1.20,
        8: 1.30,
        9: 1.40,
        10: 1.50,
    }
    adjusted_f = {}
    for number, factor in factors.items():
        if f == 0.5:
            adjusted_f[number] = 0.5
        elif f > 0.5:
            adjusted_f[number] = min(f * factor, 1.0)
        else:
            adjusted_f[number] = f * (2.0 - factor)

    m11 = _hard_pick(f)
    m12 = _hard_pick(f) if _favorite_probability(f) > 0.85 else f
    m13 = _hard_pick(f) if _favorite_probability(f) > 0.86 else f
    m14 = _hard_pick(y) if _favorite_probability(y) > 0.95 else f
    m15 = _hard_pick(y) if _favorite_probability(y) > 0.96 else f
    m16 = y if diff_yf > 0.32 else f
    m17 = _hard_pick(y) if diff_yf > 0.32 else f
    m18 = m1 if diff_yf > 0.32 else f

    m19 = v
    m20 = (y + f + v) / 3.0
    m21 = (y + v) / 2.0
    m22 = (f + v) / 2.0
    m23 = _hard_pick(m20) if _favorite_probability(m20) > 0.83 else m20
    m24 = _hard_pick(v) if _favorite_probability(v) > 0.77 else v
    m25 = v if diff_fv > 0.10 else m20

    # Formula is source of truth: fallback is Vegas, not the written three-way average.
    m26 = _hard_pick(v) if diff_fv > 0.27 else v

    m27 = m22 if diff_fv > 0.13 else m25
    m28 = m22 if diff_fv > 0.11 else m20
    m29 = m20 if diff_fv > 0.26 else m22
    m30 = (m27 + m28) / 2.0
    m31 = v if diff_fv > 0.20 else m22
    m32 = v - 0.03 if v > 0.5 else v + 0.03 if v < 0.5 else 0.5

    models = {
        1: m1,
        2: m2,
        3: m3,
        4: m4,
        **adjusted_f,
        11: m11,
        12: m12,
        13: m13,
        14: m14,
        15: m15,
        16: m16,
        17: m17,
        18: m18,
        19: m19,
        20: m20,
        21: m21,
        22: m22,
        23: m23,
        24: m24,
        25: m25,
        26: m26,
        27: m27,
        28: m28,
        29: m29,
        30: m30,
        31: m31,
        32: m32,
    }
    return _round_if(models, round_output)


def selected_legacy_ensemble_models(inputs: EnsembleInputs) -> dict[int, float]:
    all_models = legacy_ensemble_models(inputs, round_output=False)
    return {number: all_models[number] for number in (19, 20, 22, 25, 26, 29, 31, 32)}
