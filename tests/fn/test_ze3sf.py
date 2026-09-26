"""ze3sf is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ze3sf import three_step_fca


def test_ze3sf_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        three_step_fca(data=None)
