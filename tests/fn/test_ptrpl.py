"""ptrpl is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ptrpl import ripley_correction


def test_ptrpl_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ripley_correction(data=None)
