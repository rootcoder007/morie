"""idpmc is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.idpmc import idpmc


def test_idpmc_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        idpmc()
