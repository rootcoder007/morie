"""gcgcm is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gcgcm import gcgcm


def test_gcgcm_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gcgcm()
