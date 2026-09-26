"""cllvn is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.cllvn import cllvn


def test_cllvn_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        cllvn()
