"""qboIdx is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.qboIdx import qbo


def test_qboIdx_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        qbo(U30=None)
