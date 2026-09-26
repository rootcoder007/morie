"""mssmr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mssmr import smacof_replicate


def test_mssmr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        smacof_replicate(data=None)
