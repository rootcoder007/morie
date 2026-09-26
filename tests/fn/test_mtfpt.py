"""mtfpt is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mtfpt import mtfpt


def test_mtfpt_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        mtfpt()
