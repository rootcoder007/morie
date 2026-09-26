"""rndsmp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.rndsmp import rndsmp


def test_rndsmp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        rndsmp()
