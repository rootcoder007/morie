"""tmwav is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.tmwav import tmwav


def test_tmwav_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        tmwav()
