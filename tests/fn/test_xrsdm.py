"""xrsdm is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.xrsdm import sdm_ml


def test_xrsdm_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sdm_ml(data=None)
