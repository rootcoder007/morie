"""rsrtx is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.rsrtx import rsrtx


def test_rsrtx_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        rsrtx()
