"""hmcsp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hmcsp import hmcsp


def test_hmcsp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hmcsp()
