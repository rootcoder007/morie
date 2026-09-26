"""agsav is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.agsav import agsav


def test_agsav_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        agsav()
