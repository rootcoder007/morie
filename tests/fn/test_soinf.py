"""soinf is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.soinf import soinf


def test_soinf_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        soinf()
