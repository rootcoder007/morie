"""tssdm2 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.tssdm2 import tssdm2


def test_tssdm2_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        tssdm2()
