"""vgcov is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.vgcov import covariogram


def test_vgcov_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        covariogram(coords=None, values=None)
