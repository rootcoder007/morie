"""mtash is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mtash import mtash


def test_mtash_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        mtash()
