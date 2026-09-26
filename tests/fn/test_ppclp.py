"""ppclp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ppclp import ppclp


def test_ppclp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ppclp()
