"""csopt is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.csopt import csopt


def test_csopt_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        csopt()
