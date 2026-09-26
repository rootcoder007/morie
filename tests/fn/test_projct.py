"""projct is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.projct import projct


def test_projct_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        projct()
