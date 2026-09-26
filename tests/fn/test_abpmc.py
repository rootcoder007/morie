"""abpmc is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.abpmc import abpmc


def test_abpmc_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        abpmc()
