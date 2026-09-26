"""sawsp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sawsp import sawsp


def test_sawsp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sawsp()
