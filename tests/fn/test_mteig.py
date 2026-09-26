"""mteig is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mteig import mteig


def test_mteig_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        mteig()
