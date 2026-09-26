"""clssmp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.clssmp import clssmp


def test_clssmp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        clssmp()
