"""zxdps is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.zxdps import dirichlet_proc_sp


def test_zxdps_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        dirichlet_proc_sp(data=None)
