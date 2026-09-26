"""gcupw is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gcupw import gcupw


def test_gcupw_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gcupw()
