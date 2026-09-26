"""clfcm is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.clfcm import clfcm


def test_clfcm_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        clfcm()
