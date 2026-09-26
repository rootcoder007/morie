"""sveln is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sveln import elbow_spatial


def test_sveln_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        elbow_spatial(data=None)
