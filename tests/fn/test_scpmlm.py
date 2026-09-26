"""scpmlm is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.scpmlm import scpmlm


def test_scpmlm_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        scpmlm(y=None, X=None, W=None)
