"""ispol is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ispol import ispol


def test_ispol_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ispol()
