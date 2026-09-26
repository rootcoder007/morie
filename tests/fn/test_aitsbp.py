"""aitsbp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.aitsbp import aitchison_sbp_basis


def test_aitsbp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        aitchison_sbp_basis(sign=None)
