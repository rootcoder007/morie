"""mselb is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mselb import mds_elbow


def test_mselb_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        mds_elbow(X=None)
