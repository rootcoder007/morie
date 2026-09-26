"""ze2sf is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ze2sf import two_step_fca


def test_ze2sf_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        two_step_fca(data=None)
