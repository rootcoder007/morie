"""gecon is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gecon import gecon


def test_gecon_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gecon()
