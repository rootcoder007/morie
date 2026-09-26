"""ubspr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ubspr import ubspr


def test_ubspr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ubspr()
