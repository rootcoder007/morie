"""sglhd is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sglhd import sglhd


def test_sglhd_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sglhd()
