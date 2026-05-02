import numpy as np
import pandas as pd
import pytest

from pyratemaking.development import (
    BornhuetterFerguson,
    CapeCod,
    ChainLadder,
    age_to_age_factors,
    cumulative_factors,
)


def _mack_1993_triangle() -> pd.DataFrame:
    """Mack (1993) general-liability cumulative paid loss triangle."""
    data = {
        12: [357848, 352118, 290507, 310608, 443160, 396132, 440832, 359480, 376686, 344014],
        24: [
            1124788,
            1236139,
            1292306,
            1418858,
            1136350,
            1333217,
            1288463,
            1421128,
            1363294,
            np.nan,
        ],
        36: [
            1735330,
            2170033,
            2218525,
            2195047,
            2128333,
            2180715,
            2419861,
            2864498,
            np.nan,
            np.nan,
        ],
        48: [2218270, 3353322, 3235179, 3757447, 2897821, 2985752, 3483130, np.nan, np.nan, np.nan],
        60: [2745596, 3799067, 3985995, 4029929, 3402672, 3691712, np.nan, np.nan, np.nan, np.nan],
        72: [3319994, 4120063, 4132918, 4381982, 3873311, np.nan, np.nan, np.nan, np.nan, np.nan],
        84: [3466336, 4647867, 4628910, 4588268, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        96: [3606286, 4914039, 4909315, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        108: [3833515, 5339085, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        120: [3901463, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    }
    return pd.DataFrame(data, index=range(1981, 1991))


def test_age_to_age_volume_weighted_matches_mack_1993_table_2():
    tri = _mack_1993_triangle()
    f = age_to_age_factors(tri, weighted=True)
    # Mack 1993 Table 2 — volume-weighted (chain-ladder) factors.
    expected = {
        "12-24": 3.4906,
        "24-36": 1.7473,
        "36-48": 1.4574,
        "48-60": 1.1739,
        "60-72": 1.1038,
        "72-84": 1.0863,
        "84-96": 1.0539,
        "96-108": 1.0766,
        "108-120": 1.0177,
    }
    for label, want in expected.items():
        assert f.loc[label] == pytest.approx(want, abs=5e-4)


def test_chain_ladder_ultimate_for_fully_developed_ay_equals_latest():
    tri = _mack_1993_triangle()
    cl = ChainLadder(tri, tail_factor=1.0)
    ult = cl.ultimates()
    # AY 1981 reaches age 120; with no tail, ultimate == latest cumulative.
    assert ult.loc[1981, "ultimate"] == pytest.approx(3_901_463)


def test_chain_ladder_ultimate_for_1982_uses_only_108_to_120_factor():
    tri = _mack_1993_triangle()
    cl = ChainLadder(tri)
    ult = cl.ultimates()
    f = float(cl.link_factors.loc["108-120"])
    assert ult.loc[1982, "ultimate"] == pytest.approx(5_339_085 * f, rel=1e-9)


def test_chain_ladder_ultimate_for_1990_uses_full_chain():
    tri = _mack_1993_triangle()
    cl = ChainLadder(tri)
    ult = cl.ultimates()
    full_cdf = float(cl.cdf.loc[12])
    assert ult.loc[1990, "ultimate"] == pytest.approx(344_014 * full_cdf, rel=1e-9)


def test_cumulative_factors_apply_tail_at_end():
    f = pd.Series({"12-24": 2.0, "24-36": 1.5})
    cdf = cumulative_factors(f, tail=1.1)
    assert cdf.loc[36] == pytest.approx(1.1)
    assert cdf.loc[24] == pytest.approx(1.5 * 1.1)
    assert cdf.loc[12] == pytest.approx(2.0 * 1.5 * 1.1)


def test_chain_ladder_simple_triangle_basic_check():
    tri = pd.DataFrame(
        {
            12: [100, 110, 120],
            24: [150, 160, np.nan],
            36: [170, np.nan, np.nan],
        },
        index=[2018, 2019, 2020],
    )
    cl = ChainLadder(tri)
    f12_24 = (150 + 160) / (100 + 110)
    f24_36 = 170 / 150
    assert cl.link_factors.loc["12-24"] == pytest.approx(f12_24)
    assert cl.link_factors.loc["24-36"] == pytest.approx(f24_36)
    ult = cl.ultimates()
    assert ult.loc[2018, "ultimate"] == pytest.approx(170)
    assert ult.loc[2019, "ultimate"] == pytest.approx(160 * f24_36)
    assert ult.loc[2020, "ultimate"] == pytest.approx(120 * f12_24 * f24_36)


def test_bornhuetter_ferguson_matches_textbook_formula():
    tri = pd.DataFrame(
        {
            12: [100, 110, 120],
            24: [150, 160, np.nan],
            36: [170, np.nan, np.nan],
        },
        index=[2018, 2019, 2020],
    )
    a_priori = pd.Series({2018: 200, 2019: 200, 2020: 200})
    bf = BornhuetterFerguson(tri, a_priori_ultimate=a_priori)
    out = bf.ultimates()
    cdf = bf.cdf
    # AY 2020 reported at age 12 with cdf-to-ult = ?
    pct = 1.0 / float(cdf.loc[12])
    expected_2020 = 120 + 200 * (1 - pct)
    assert out.loc[2020, "ultimate"] == pytest.approx(expected_2020)


def test_cape_cod_elr_consistent_with_definition():
    tri = pd.DataFrame(
        {
            12: [100, 110, 120],
            24: [150, 160, np.nan],
            36: [170, np.nan, np.nan],
        },
        index=[2018, 2019, 2020],
    )
    used_premium = pd.Series({2018: 250, 2019: 260, 2020: 270})
    cc = CapeCod(tri, used_premium=used_premium)
    elr = cc.expected_loss_ratio
    # ELR = sum(reported) / sum(premium * pct_reported)
    cdf = cc.cdf
    pct12 = 1 / float(cdf.loc[12])
    pct24 = 1 / float(cdf.loc[24])
    pct36 = 1 / float(cdf.loc[36])
    num = 170 + 160 + 120
    den = 250 * pct36 + 260 * pct24 + 270 * pct12
    assert elr == pytest.approx(num / den)


def test_chain_ladder_repr_is_concise():
    tri = pd.DataFrame({12: [100, 110], 24: [150, np.nan]}, index=[2019, 2020])
    text = repr(ChainLadder(tri))
    assert "ChainLadder" in text
    assert "tail_factor=1.0000" in text
