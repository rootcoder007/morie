"""markg is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.markg import markg


def test_markg_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        markg()
