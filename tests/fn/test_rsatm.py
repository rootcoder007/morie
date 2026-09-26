"""rsatm is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.rsatm import rsatm


def test_rsatm_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        rsatm()
