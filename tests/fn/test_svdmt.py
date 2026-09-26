"""svdmt is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.svdmt import dim_test


def test_svdmt_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        dim_test(data=None)
