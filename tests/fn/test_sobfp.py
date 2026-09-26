"""sobfp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sobfp import sobfp


def test_sobfp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sobfp()
