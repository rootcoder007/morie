"""gdlfe is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gdlfe import gdlfe


def test_gdlfe_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gdlfe()
