"""clstal is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.clstal import clustalo


def test_clstal_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        clustalo(sequences=None)
