"""scpois is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.scpois import scpois


def test_scpois_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        scpois(y=None, X=None, W=None)
