"""zsstn is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.zsstn import st_cov_nonsep


def test_zsstn_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        st_cov_nonsep(data=None)
