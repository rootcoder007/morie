"""rsalb is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.rsalb import rsalb


def test_rsalb_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        rsalb()
