"""vmpsc is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.vmpsc import vmpsc


def test_vmpsc_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        vmpsc()
