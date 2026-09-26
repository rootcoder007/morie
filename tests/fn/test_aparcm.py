"""aparcm is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.aparcm import aparch_dge


def test_aparcm_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        aparch_dge(x=None, delta=None)
