"""ppann is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ppann import ppann


def test_ppann_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ppann()
