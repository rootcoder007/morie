"""nmpol is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.nmpol import leg_polarity


def test_nmpol_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        leg_polarity(data=None)
