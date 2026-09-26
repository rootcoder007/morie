"""csjrn is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.csjrn import csjrn


def test_csjrn_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        csjrn()
