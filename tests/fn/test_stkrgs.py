"""stkrgs is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.stkrgs import stkrgs


def test_stkrgs_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        stkrgs()
