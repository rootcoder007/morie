"""ulrnir is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ulrnir import u_learner


def test_ulrnir_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        u_learner(y=None, D=None, X=None)
