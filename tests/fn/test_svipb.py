"""svipb is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.svipb import ideal_point_bayes


def test_svipb_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ideal_point_bayes(data=None)
