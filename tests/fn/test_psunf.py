"""psunf is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.psunf import psunf


def test_psunf_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        psunf()
