"""spcsim2 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.spcsim2 import spcsim2


def test_spcsim2_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        spcsim2()
