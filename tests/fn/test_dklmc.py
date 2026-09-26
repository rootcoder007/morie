"""dklmc is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.dklmc import dklmc


def test_dklmc_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        dklmc()
