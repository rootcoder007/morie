"""mdsta is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mdsta import mdsta


def test_mdsta_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        mdsta()
