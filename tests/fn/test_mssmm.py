"""mssmm is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mssmm import smacof_missing


def test_mssmm_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        smacof_missing(data=None)
