"""zsrbf is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.zsrbf import rbf_multiquad


def test_zsrbf_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        rbf_multiquad(data=None)
