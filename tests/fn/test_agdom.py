"""agdom is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.agdom import agdom


def test_agdom_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        agdom()
