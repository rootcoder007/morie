"""cksim is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.cksim import cksim


def test_cksim_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        cksim()
