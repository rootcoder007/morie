"""plbdn is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.plbdn import plbdn


def test_plbdn_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        plbdn()
