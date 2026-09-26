"""ppnyn is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ppnyn import ppnyn


def test_ppnyn_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ppnyn()
