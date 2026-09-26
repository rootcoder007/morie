"""zsrbt is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.zsrbt import rbf_thinplate


def test_zsrbt_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        rbf_thinplate(data=None)
