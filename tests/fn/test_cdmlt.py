"""cdmlt is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.cdmlt import cdmlt


def test_cdmlt_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        cdmlt()
