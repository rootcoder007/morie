"""spcani is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.spcani import spcani


def test_spcani_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        spcani()
