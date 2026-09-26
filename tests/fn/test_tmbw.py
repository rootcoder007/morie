"""tmbw is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.tmbw import tmbw


def test_tmbw_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        tmbw()
