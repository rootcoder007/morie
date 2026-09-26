"""rstcw is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.rstcw import rstcw


def test_rstcw_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        rstcw()
