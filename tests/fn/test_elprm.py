"""elprm is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.elprm import elprm


def test_elprm_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        elprm()
