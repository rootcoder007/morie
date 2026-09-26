"""gdimm is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gdimm import gdimm


def test_gdimm_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gdimm()
