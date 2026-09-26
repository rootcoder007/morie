"""psmix is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.psmix import psmix


def test_psmix_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        psmix()
