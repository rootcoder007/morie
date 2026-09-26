"""ummec is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ummec import ummec


def test_ummec_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ummec()
