"""elsal is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.elsal import elsal


def test_elsal_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        elsal()
