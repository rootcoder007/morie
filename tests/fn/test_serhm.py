"""serhm is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.serhm import serhm


def test_serhm_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        serhm()
