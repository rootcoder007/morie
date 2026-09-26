"""zetwn is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.zetwn import townsend_index


def test_zetwn_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        townsend_index(data=None)
