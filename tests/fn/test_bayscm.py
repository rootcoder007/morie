"""bayscm is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.bayscm import bayes_c_pi


def test_bayscm_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        bayes_c_pi(y=None, M=None, pi=None)
