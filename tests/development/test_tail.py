import numpy as np
import pandas as pd
import pytest

from pyratemaking.development.tail import (
    bondy_tail,
    exponential_decay_tail,
    power_curve_tail,
    select_tail,
    sherman_tail,
)


def _decaying_link_factors():
    ages = [12, 24, 36, 48, 60, 72, 84, 96]
    ldfs = 1 + 2.0 / np.array(ages, dtype=float) ** 1.5
    return pd.Series(ldfs, index=[f"{ages[i]}-{ages[i] + 12}" for i in range(len(ages))])


def test_bondy_returns_last_link_factor():
    f = pd.Series([2.0, 1.5, 1.2], index=["12-24", "24-36", "36-48"])
    tail, info = bondy_tail(f)
    assert tail == pytest.approx(1.2)
    assert info.loc[0, "method"] == "bondy"


def test_bondy_modified_with_power_two():
    f = pd.Series([1.5, 1.2, 1.1], index=["12-24", "24-36", "36-48"])
    tail, _ = bondy_tail(f, power=2)
    assert tail == pytest.approx(1.1**2)


def test_sherman_recovers_inverse_power_form():
    f = _decaying_link_factors()
    tail, info = sherman_tail(f, periods_to_extend=200)
    assert tail > 1.0
    assert tail < 1.5
    # The fit should recover the exponent close to 1.5
    assert info.loc[0, "d"] == pytest.approx(1.5, abs=1e-2)


def test_power_curve_recovers_exponent():
    f = _decaying_link_factors()
    _, info = power_curve_tail(f, periods_to_extend=50)
    assert info.loc[0, "b"] == pytest.approx(1.5, abs=1e-2)


def test_exponential_decay_returns_finite_tail():
    f = _decaying_link_factors()
    tail, _info = exponential_decay_tail(f, periods_to_extend=20)
    assert np.isfinite(tail)
    assert tail >= 1.0


def test_select_tail_returns_one_row_per_method():
    f = _decaying_link_factors()
    out = select_tail(f, methods=("bondy", "sherman", "power", "exp_decay"))
    assert set(out.index) == {"bondy", "sherman", "power", "exp_decay"}


def test_select_tail_unknown_method_raises():
    f = pd.Series([1.5], index=["12-24"])
    with pytest.raises(ValueError, match="unknown tail"):
        select_tail(f, methods=("not_a_method",))


def test_sherman_with_no_excess_returns_unity():
    f = pd.Series([1.0, 1.0, 1.0], index=["12-24", "24-36", "36-48"])
    tail, _ = sherman_tail(f)
    assert tail == 1.0
