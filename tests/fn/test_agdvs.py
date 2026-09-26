"""agdvs is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.agdvs import agdvs


def test_agdvs_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        agdvs()
