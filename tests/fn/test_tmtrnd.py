"""tmtrnd is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.tmtrnd import tmtrnd


def test_tmtrnd_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        tmtrnd()
