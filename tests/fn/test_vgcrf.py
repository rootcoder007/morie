"""vgcrf is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.vgcrf import correlogram


def test_vgcrf_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        correlogram(data=None)
