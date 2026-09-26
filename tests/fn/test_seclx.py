"""seclx is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.seclx import seclx


def test_seclx_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        seclx()
