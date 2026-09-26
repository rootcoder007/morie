"""gctmp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gctmp import gctmp


def test_gctmp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gctmp()
