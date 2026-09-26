"""sawis is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sawis import sawis


def test_sawis_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sawis()
