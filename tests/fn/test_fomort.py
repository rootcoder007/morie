"""fomort is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.fomort import fomort


def test_fomort_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        fomort()
