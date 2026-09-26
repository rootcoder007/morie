"""gdhhs is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gdhhs import gdhhs


def test_gdhhs_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gdhhs()
