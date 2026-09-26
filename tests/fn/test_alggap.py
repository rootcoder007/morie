"""alggap is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.alggap import algebraic_connectivity


def test_alggap_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        algebraic_connectivity(G=None)
