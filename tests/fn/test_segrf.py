"""segrf is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.segrf import segrf


def test_segrf_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        segrf()
