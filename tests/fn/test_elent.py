"""elent is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.elent import elent


def test_elent_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        elent()
