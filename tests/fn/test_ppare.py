"""ppare is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ppare import ppare


def test_ppare_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ppare()
