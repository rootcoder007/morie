"""tssmf is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.tssmf import tssmf


def test_tssmf_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        tssmf()
