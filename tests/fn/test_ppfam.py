"""ppfam is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ppfam import ppfam


def test_ppfam_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ppfam()
