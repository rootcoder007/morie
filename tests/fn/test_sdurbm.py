"""sdurbm is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sdurbm import spatial_durbin_model


def test_sdurbm_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        spatial_durbin_model(y=None, X=None, W=None)
