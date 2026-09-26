"""baysbm is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.baysbm import bayes_b_marker


def test_baysbm_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        bayes_b_marker(y=None, M=None, pi=None)
