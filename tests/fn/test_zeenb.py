"""zeenb is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.zeenb import ecological_nb


def test_zeenb_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ecological_nb(data=None)
