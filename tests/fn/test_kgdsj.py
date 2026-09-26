"""kgdsj is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.kgdsj import disjunctive_kriging


def test_kgdsj_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        disjunctive_kriging(values=None, x=None)
