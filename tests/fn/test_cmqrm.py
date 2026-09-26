"""cmqrm is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.cmqrm import cmqrm


def test_cmqrm_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        cmqrm()
