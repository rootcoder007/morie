"""soper is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.soper import soper


def test_soper_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        soper()
