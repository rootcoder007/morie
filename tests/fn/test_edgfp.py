"""edgfp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.edgfp import edgfp


def test_edgfp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        edgfp()
