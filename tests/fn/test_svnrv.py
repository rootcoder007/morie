"""svnrv is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.svnrv import normal_vector


def test_svnrv_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        normal_vector(data=None)
