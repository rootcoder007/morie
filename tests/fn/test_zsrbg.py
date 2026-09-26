"""zsrbg is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.zsrbg import rbf_gaussian


def test_zsrbg_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        rbf_gaussian(data=None)
