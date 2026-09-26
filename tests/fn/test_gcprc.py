"""gcprc is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gcprc import gcprc


def test_gcprc_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gcprc()
