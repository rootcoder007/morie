"""mcsbs is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mcsbs import mcsbs


def test_mcsbs_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        mcsbs()
