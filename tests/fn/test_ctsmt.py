"""ctsmt is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ctsmt import ctsmt


def test_ctsmt_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ctsmt()
