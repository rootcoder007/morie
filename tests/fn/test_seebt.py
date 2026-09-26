"""seebt is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.seebt import seebt


def test_seebt_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        seebt()
