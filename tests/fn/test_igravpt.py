"""igravpt is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.igravpt import igravpt


def test_igravpt_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        igravpt(mass=None, dist=None)
