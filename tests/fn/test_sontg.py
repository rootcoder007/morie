"""sontg is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sontg import sontg


def test_sontg_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sontg()
