"""Tests for morie.fn.sirdm -- SIR with age-structured demographics."""

import pytest

from morie.fn.sirdm import sir_age_demographics


class TestSIRDM:
    def test_runs(self):
        res = sir_age_demographics()
        assert res.model == "SIR-AGE"
        assert len(res.t) == 1000

    def test_r0_positive(self):
        res = sir_age_demographics()
        assert res.R0 > 0

    def test_invalid_gamma(self):
        with pytest.raises(ValueError):
            sir_age_demographics(gamma=-1)

    def test_r0_is_the_spectral_radius_of_beta_over_gamma(self):
        import math

        B = [[0.4, 0.1], [0.2, 0.3]]
        g = 0.2
        tr = (B[0][0] + B[1][1]) / g
        det = (B[0][0] * B[1][1] - B[0][1] * B[1][0]) / g**2
        rho = tr / 2 + math.sqrt(tr * tr / 4 - det)
        res = sir_age_demographics(beta_matrix=B, gamma=g, t_max=10.0, n_steps=50)
        assert pytest.approx(rho, rel=1e-12) == res.R0
        assert res.S[0] == pytest.approx(0.98, rel=1e-14)
        assert res.I[0] == pytest.approx(0.01, rel=1e-14)
