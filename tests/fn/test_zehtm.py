"""zehtm is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.zehtm import hotspot_map


def test_zehtm_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hotspot_map(data=None)
