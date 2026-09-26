"""zsstc is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.zsstc import st_cov_sep


def test_zsstc_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        st_cov_sep(data=None)
