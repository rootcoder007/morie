"""spcext is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.spcext import spcext


def test_spcext_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        spcext()
