"""Test lll_reduce."""

import numpy as np

from morie.fn._containers import DescriptiveResult
from morie.fn.lll import lll_reduce


class TestLllReduce:
    def test_basic(self):
        basis = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 2]], dtype=float)
        result = lll_reduce(basis=basis)
        assert result.extra["dimension"] is not None

    def test_output_type(self):
        basis = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 2]], dtype=float)
        result = lll_reduce(basis=basis)
        assert isinstance(result, DescriptiveResult)

    def test_reduced_basis_in_extra(self):
        basis = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 2]], dtype=float)
        result = lll_reduce(basis=basis)
        assert "reduced_basis" in result.extra
