"""sarbic is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sarbic import sarbic


def test_sarbic_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sarbic(ll=None, k=None, n=None)
