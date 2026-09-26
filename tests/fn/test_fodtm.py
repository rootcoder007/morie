"""fodtm is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.fodtm import fodtm


def test_fodtm_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        fodtm()
