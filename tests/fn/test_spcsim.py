"""spcsim is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.spcsim import spcsim


def test_spcsim_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        spcsim()
