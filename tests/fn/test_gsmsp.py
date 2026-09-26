"""gsmsp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gsmsp import gsmsp


def test_gsmsp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gsmsp()
