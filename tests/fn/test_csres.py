"""csres is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.csres import csres


def test_csres_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        csres()
