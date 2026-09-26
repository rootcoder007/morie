"""cdlmc is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.cdlmc import cdlmc


def test_cdlmc_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        cdlmc()
