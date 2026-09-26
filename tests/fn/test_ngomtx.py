"""ngomtx is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ngomtx import next_generation_matrix


def test_ngomtx_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        next_generation_matrix(FV_decomposition=None)
