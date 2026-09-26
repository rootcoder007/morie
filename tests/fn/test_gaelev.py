"""gaelev is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gaelev import gaelev


def test_gaelev_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gaelev()
