"""ghlam is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ghlam import ghlam


def test_ghlam_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ghlam()
