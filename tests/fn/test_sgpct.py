"""sgpct is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sgpct import sgpct


def test_sgpct_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sgpct()
