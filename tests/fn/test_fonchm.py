"""fonchm is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.fonchm import fonchm


def test_fonchm_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        fonchm()
