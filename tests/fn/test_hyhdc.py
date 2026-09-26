"""hyhdc is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hyhdc import hyhdc


def test_hyhdc_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hyhdc()
