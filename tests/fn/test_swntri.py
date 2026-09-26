"""swntri is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.swntri import swntri


def test_swntri_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        swntri(W=None)
