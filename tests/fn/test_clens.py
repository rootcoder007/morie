"""clens is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.clens import clens


def test_clens_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        clens()
