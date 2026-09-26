"""seclf is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.seclf import seclf


def test_seclf_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        seclf()
