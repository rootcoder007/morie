"""zsste is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.zsste import st_cressie_huang


def test_zsste_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        st_cressie_huang(data=None)
