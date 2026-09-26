"""msans is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.msans import mds_aniso


def test_msans_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        mds_aniso(X=None)
