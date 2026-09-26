"""sgspd is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sgspd import space_deformation


def test_sgspd_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        space_deformation()
