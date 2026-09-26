"""swcond is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.swcond import swcond


def test_swcond_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        swcond(W=None)
