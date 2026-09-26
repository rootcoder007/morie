"""jkrand is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.jkrand import jackknife_repl


def test_jkrand_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        jackknife_repl(theta_replicates=None)
