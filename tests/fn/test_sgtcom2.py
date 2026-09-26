"""sgtcom2 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sgtcom2 import sgt_communicability_matrix


def test_sgtcom2_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sgt_communicability_matrix(A=None)
