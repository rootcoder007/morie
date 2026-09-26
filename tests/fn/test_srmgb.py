"""srmgb is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.srmgb import srmgb


def test_srmgb_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        srmgb()
