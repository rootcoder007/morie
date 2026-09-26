"""ctband is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ctband import ctband


def test_ctband_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ctband()
