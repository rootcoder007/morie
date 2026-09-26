"""ppjcq is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ppjcq import ppjcq


def test_ppjcq_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ppjcq()
