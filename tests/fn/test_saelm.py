"""saelm is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.saelm import saelm


def test_saelm_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        saelm()
