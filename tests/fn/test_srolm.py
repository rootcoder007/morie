"""srolm is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.srolm import srolm


def test_srolm_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        srolm()
