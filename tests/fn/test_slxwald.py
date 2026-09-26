"""slxwald is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.slxwald import slxwald


def test_slxwald_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        slxwald(theta=None, se_theta=None)
