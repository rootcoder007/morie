"""mdctr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mdctr import mdctr


def test_mdctr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        mdctr()
