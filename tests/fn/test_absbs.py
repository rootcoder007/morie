"""absbs is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.absbs import absbs


def test_absbs_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        absbs()
