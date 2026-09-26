"""enplm is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.enplm import enplm


def test_enplm_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        enplm()
