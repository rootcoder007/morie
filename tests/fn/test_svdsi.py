"""svdsi is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.svdsi import svdsi


def test_svdsi_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        svdsi()
