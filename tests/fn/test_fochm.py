"""fochm is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.fochm import fochm


def test_fochm_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        fochm()
