"""tssgn is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.tssgn import tssgn


def test_tssgn_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        tssgn()
