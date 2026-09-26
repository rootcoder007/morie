"""nbmrn is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.nbmrn import nbmrn


def test_nbmrn_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        nbmrn()
