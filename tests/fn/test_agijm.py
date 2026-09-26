"""agijm is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.agijm import agijm


def test_agijm_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        agijm()
