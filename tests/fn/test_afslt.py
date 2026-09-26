"""afslt is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.afslt import afslt


def test_afslt_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        afslt()
