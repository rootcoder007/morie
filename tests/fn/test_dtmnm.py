"""dtmnm is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.dtmnm import dtmnm


def test_dtmnm_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        dtmnm()
