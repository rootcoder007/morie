"""md3dp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.md3dp import md3dp


def test_md3dp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        md3dp()
