"""elrun is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.elrun import elrun


def test_elrun_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        elrun()
