"""concav is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.concav import concav


def test_concav_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        concav()
