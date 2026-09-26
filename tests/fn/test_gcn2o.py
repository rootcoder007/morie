"""gcn2o is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gcn2o import gcn2o


def test_gcn2o_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gcn2o()
