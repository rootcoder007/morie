"""csbt is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.csbt import csbt


def test_csbt_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        csbt()
