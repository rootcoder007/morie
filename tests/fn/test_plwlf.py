"""plwlf is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.plwlf import plwlf


def test_plwlf_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        plwlf()
