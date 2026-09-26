"""slearn is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.slearn import s_learner


def test_slearn_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        s_learner(y=None, D=None, X=None)
