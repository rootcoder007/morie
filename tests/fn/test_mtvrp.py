"""mtvrp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mtvrp import mtvrp


def test_mtvrp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        mtvrp()
