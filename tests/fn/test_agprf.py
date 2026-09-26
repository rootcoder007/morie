"""agprf is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.agprf import agprf


def test_agprf_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        agprf()
