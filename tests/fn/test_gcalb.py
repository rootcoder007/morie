"""gcalb is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gcalb import gcalb


def test_gcalb_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gcalb()
