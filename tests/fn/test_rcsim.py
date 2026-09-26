"""rcsim is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.rcsim import rcsim


def test_rcsim_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        rcsim()
