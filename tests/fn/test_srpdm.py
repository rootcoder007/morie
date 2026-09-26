"""srpdm is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.srpdm import srpdm


def test_srpdm_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        srpdm()
