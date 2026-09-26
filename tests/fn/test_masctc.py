"""masctc is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.masctc import masctc


def test_masctc_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        masctc()
