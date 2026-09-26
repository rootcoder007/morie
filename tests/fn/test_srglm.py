"""srglm is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.srglm import srglm


def test_srglm_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        srglm()
