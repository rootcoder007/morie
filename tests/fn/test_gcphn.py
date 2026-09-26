"""gcphn is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gcphn import gcphn


def test_gcphn_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gcphn()
