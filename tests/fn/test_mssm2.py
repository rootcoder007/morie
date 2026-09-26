"""mssm2 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mssm2 import smacof_2d


def test_mssm2_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        smacof_2d(data=None)
