"""ghgrv is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ghgrv import ghgrv


def test_ghgrv_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ghgrv()
