"""sdmmod is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sdmmod import spatial_durbin


def test_sdmmod_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        spatial_durbin(y=None, X=None, W=None)
