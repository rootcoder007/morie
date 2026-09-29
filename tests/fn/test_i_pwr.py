"""Tests for morie.fn.i_pwr: equals the one-numerator-df noncentral-F power."""

from morie.fn.i_pwr import calculate_interaction_power
from morie.fn.pwr_av import power_anova


def test_single_df_equals_two_group_anova():
    # df = (1, N - 2), ncp = f^2 N is the two-group ANOVA with n = N / 2 per group
    for N, f in ((60, 0.2), (200, 0.15), (40, 0.5)):
        assert abs(calculate_interaction_power(N, effect_size=f) - power_anova(n=N / 2, k=2, f=f)) < 1e-12


def test_power_increases_with_n():
    assert calculate_interaction_power(100) < calculate_interaction_power(300) < 1.0
