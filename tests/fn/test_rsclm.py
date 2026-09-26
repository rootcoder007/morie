"""rsclm is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.rsclm import rsclm


def test_rsclm_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        rsclm()
