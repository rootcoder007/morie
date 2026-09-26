"""gcstd is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gcstd import gcstd


def test_gcstd_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gcstd()
