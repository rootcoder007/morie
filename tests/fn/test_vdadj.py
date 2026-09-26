"""vdadj is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.vdadj import vdadj


def test_vdadj_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        vdadj()
