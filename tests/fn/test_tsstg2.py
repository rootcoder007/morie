"""tsstg2 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.tsstg2 import tsstg2


def test_tsstg2_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        tsstg2()
