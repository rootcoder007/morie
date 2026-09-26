"""dtbic is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.dtbic import dtbic


def test_dtbic_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        dtbic()
