"""mtlvy is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mtlvy import mtlvy


def test_mtlvy_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        mtlvy()
