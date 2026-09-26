"""tbaniso is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.tbaniso import tbaniso


def test_tbaniso_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        tbaniso()
