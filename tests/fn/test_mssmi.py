"""mssmi is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mssmi import smacof_indiv


def test_mssmi_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        smacof_indiv(data=None)
