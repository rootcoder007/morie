"""mcimp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mcimp import mcimp


def test_mcimp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        mcimp()
