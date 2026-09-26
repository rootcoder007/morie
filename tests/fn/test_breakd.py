"""breakd is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.breakd import breakdown_point


def test_breakd_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        breakdown_point(estimator=None, n=None)
