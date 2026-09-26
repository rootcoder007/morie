"""cdcl is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.cdcl import cdcl


def test_cdcl_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        cdcl(cnf=None)
