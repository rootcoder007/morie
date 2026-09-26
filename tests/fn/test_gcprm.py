"""gcprm is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gcprm import gcprm


def test_gcprm_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gcprm()
