"""swsym is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.swsym import swsym


def test_swsym_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        swsym(W=None)
